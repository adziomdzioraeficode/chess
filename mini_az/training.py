"""Replay buffer, collation, train_step, and data augmentation."""

import math
import os
import random
import gzip
import pickle
import shutil
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .network import ChessNet


@dataclass
class Sample:
    board_planes: np.ndarray
    moves_fs: np.ndarray
    moves_ts: np.ndarray
    moves_pr: np.ndarray
    target_pi: np.ndarray
    z: float
    wdl: np.ndarray  # [P(win), P(draw), P(loss)] target — 3 floats
    plies_left: float = 0.0

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "plies_left" not in self.__dict__:
            self.plies_left = 0.0


class ReplayBuffer:
    def __init__(self, maxlen=200_000):
        self.maxlen = maxlen
        self.data: list[Sample] = []
        self.pos = 0

    def resize(self, new_maxlen: int) -> "ReplayBuffer":
        new_maxlen = int(new_maxlen)
        if new_maxlen == self.maxlen:
            return self

        if len(self.data) <= new_maxlen:
            self.maxlen = new_maxlen
            self.pos = self.pos % max(1, new_maxlen)
            return self

        n = len(self.data)
        if n == self.maxlen:
            ordered = self.data[self.pos:] + self.data[:self.pos]
        else:
            ordered = self.data[:]

        ordered = ordered[-new_maxlen:]
        self.data = ordered
        self.maxlen = new_maxlen
        self.pos = 0 if len(self.data) < self.maxlen else 0
        return self

    def add_game(self, samples: list[Sample]):
        for s in samples:
            if len(self.data) < self.maxlen:
                self.data.append(s)
            else:
                self.data[self.pos] = s
                self.pos = (self.pos + 1) % self.maxlen

    def __len__(self):
        return len(self.data)

    def sample_batch(self, batch_size: int) -> list[Sample]:
        n = len(self.data)
        idxs = np.random.randint(0, n, size=batch_size)
        return [self.data[i] for i in idxs]

    def _sample_indices_uniform(self, n: int) -> np.ndarray:
        return np.random.randint(0, len(self.data), size=n)

    def _sample_indices_decisive(self, n: int, threshold: float = 0.3) -> np.ndarray:
        """Sample indices biased towards decisive games (|z| > threshold).

        Uses rejection sampling to avoid building a full weight array over 500k
        samples. Falls back to uniform if fewer than n/2 decisive found.
        """
        if n <= 0 or len(self.data) == 0:
            return np.array([], dtype=np.int64)
        found: list[int] = []
        tries_left = max(64, 16 * n)
        size = len(self.data)
        while len(found) < n and tries_left > 0:
            idx = int(np.random.randint(0, size))
            if abs(float(getattr(self.data[idx], "z", 0.0))) > threshold:
                found.append(idx)
            tries_left -= 1
        if len(found) < n // 2:
            # Too few decisive samples — fall back to uniform
            return np.random.randint(0, size, size=n).astype(np.int64)
        if len(found) < n:
            extra = np.random.choice(np.asarray(found, dtype=np.int64), size=n - len(found), replace=True)
            return np.concatenate([np.asarray(found, dtype=np.int64), extra])
        return np.asarray(found, dtype=np.int64)

    def _sample_indices_recent(self, n: int, window: int) -> np.ndarray:
        size = len(self.data)
        if size == 0:
            return np.array([], dtype=np.int64)

        window = int(min(max(1, window), size))

        if size < self.maxlen:
            start = size - window
            return start + np.random.randint(0, window, size=n)

        newest = (self.pos - 1) % self.maxlen
        offs = np.random.randint(0, window, size=n)
        idxs = (newest - offs) % self.maxlen
        return idxs.astype(np.int64)

    def _sample_indices_sharp(self, n: int, threshold: float) -> np.ndarray:
        if n <= 0 or len(self.data) == 0:
            return np.array([], dtype=np.int64)
        found: list[int] = []
        tries_left = max(64, 16 * n)
        size = len(self.data)

        while len(found) < n and tries_left > 0:
            idx = int(np.random.randint(0, size))
            if abs(float(getattr(self.data[idx], "z", 0.0))) >= threshold:
                found.append(idx)
            tries_left -= 1

        if not found:
            return np.array([], dtype=np.int64)
        if len(found) < n:
            extra = np.random.choice(np.asarray(found, dtype=np.int64), size=n - len(found), replace=True)
            return np.concatenate([np.asarray(found, dtype=np.int64), extra])
        return np.asarray(found, dtype=np.int64)

    def sample_batch_mixed(
        self,
        batch_size: int,
        recent_frac: float = 0.5,
        recent_window: int = 200_000,
        sharp_frac: float = 0.0,
        sharp_threshold: float = 0.35,
        decisive_frac: float = 0.0,
    ) -> list[Sample]:
        size = len(self.data)
        if size == 0:
            return []

        recent_frac = float(np.clip(recent_frac, 0.0, 1.0))
        sharp_frac = float(np.clip(sharp_frac, 0.0, 1.0))
        decisive_frac = float(np.clip(decisive_frac, 0.0, 1.0))

        n_sharp = int(round(batch_size * sharp_frac))
        n_decisive = int(round(batch_size * decisive_frac))
        n_remaining = batch_size - n_sharp - n_decisive
        n_recent = int(round(n_remaining * recent_frac))
        n_all = n_remaining - n_recent

        idxs = []
        if n_sharp > 0:
            sharp_idxs = self._sample_indices_sharp(n_sharp, sharp_threshold)
            idxs.append(sharp_idxs)
            n_missing = n_sharp - int(sharp_idxs.size)
            if n_missing > 0:
                n_recent += n_missing
        if n_decisive > 0:
            dec_idxs = self._sample_indices_decisive(n_decisive)
            idxs.append(dec_idxs)
        if n_recent > 0:
            idxs.append(self._sample_indices_recent(n_recent, recent_window))
        if n_all > 0:
            idxs.append(self._sample_indices_uniform(n_all))

        idxs = [x for x in idxs if x.size > 0]
        if not idxs:
            return []
        idxs = np.concatenate(idxs) if len(idxs) > 1 else idxs[0]
        np.random.shuffle(idxs)

        return [self.data[int(i)] for i in idxs]

    def dump(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        bak = f"{path}.bak"
        if os.path.exists(path):
            try:
                os.replace(path, bak)
            except Exception:
                shutil.copy2(path, bak)
        with gzip.open(tmp, "wb") as f:
            pickle.dump(
                {"maxlen": int(self.maxlen), "pos": int(self.pos), "data": self.data},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "ReplayBuffer":
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
        rb = ReplayBuffer(maxlen=int(obj["maxlen"]))
        rb.data = obj["data"]
        rb.pos = int(obj.get("pos", 0)) % max(1, rb.maxlen)
        return rb


def flip_sample_lr(s: Sample) -> Sample:
    bp = s.board_planes[:, :, ::-1].copy()
    # Swap kingside ↔ queenside castling planes (mirroring flips the board files)
    from .encoding import PL_MY_OO, PL_MY_OOO, PL_OPP_OO, PL_OPP_OOO
    bp[PL_MY_OO], bp[PL_MY_OOO] = bp[PL_MY_OOO].copy(), bp[PL_MY_OO].copy()
    bp[PL_OPP_OO], bp[PL_OPP_OOO] = bp[PL_OPP_OOO].copy(), bp[PL_OPP_OO].copy()
    fs = (s.moves_fs ^ 7).copy()
    ts = (s.moves_ts ^ 7).copy()
    return Sample(bp, fs, ts, s.moves_pr.copy(), s.target_pi.copy(), s.z, s.wdl.copy(), s.plies_left)


def collate(samples: List[Sample], device: str):
    # --- Augmentation: vectorised LR flip decision ---
    from .encoding import PL_MY_OO, PL_MY_OOO, PL_OPP_OO, PL_OPP_OOO

    B = len(samples)
    flip_mask = np.random.random(B) < 0.5

    # Pre-compute Lmax in one pass
    lengths = np.array([s.moves_fs.shape[0] for s in samples], dtype=np.int32)
    Lmax = int(lengths.max()) if B > 0 else 1

    # Pre-allocate numpy arrays (zero-filled = correct padding)
    bp_arr = np.empty((B, samples[0].board_planes.shape[0], 8, 8), dtype=np.float32)
    fs_arr = np.zeros((B, Lmax), dtype=np.int64)
    ts_arr = np.zeros((B, Lmax), dtype=np.int64)
    pr_arr = np.zeros((B, Lmax), dtype=np.int64)
    mask_arr = np.zeros((B, Lmax), dtype=np.bool_)
    pi_arr = np.zeros((B, Lmax), dtype=np.float32)
    z_arr = np.empty(B, dtype=np.float32)
    wdl_arr = np.empty((B, 3), dtype=np.float32)
    pl_arr = np.empty(B, dtype=np.float32)

    for i, s in enumerate(samples):
        L = int(lengths[i])
        if flip_mask[i]:
            # Inline LR flip: mirror files, swap castling planes
            bp = s.board_planes[:, :, ::-1]
            bp_arr[i] = bp
            bp_arr[i, PL_MY_OO], bp_arr[i, PL_MY_OOO] = bp[PL_MY_OOO].copy(), bp[PL_MY_OO].copy()
            bp_arr[i, PL_OPP_OO], bp_arr[i, PL_OPP_OOO] = bp[PL_OPP_OOO].copy(), bp[PL_OPP_OO].copy()
            fs_arr[i, :L] = s.moves_fs ^ 7
            ts_arr[i, :L] = s.moves_ts ^ 7
        else:
            bp_arr[i] = s.board_planes
            fs_arr[i, :L] = s.moves_fs
            ts_arr[i, :L] = s.moves_ts
        pr_arr[i, :L] = s.moves_pr
        mask_arr[i, :L] = True
        pi_arr[i, :L] = s.target_pi
        z_arr[i] = s.z
        wdl_arr[i] = s.wdl
        pl_arr[i] = float(getattr(s, "plies_left", 0.0))

    # Single bulk conversion numpy → torch (one .to(device) per tensor)
    boards = torch.from_numpy(bp_arr).to(device)
    fs = torch.from_numpy(fs_arr).to(device)
    ts = torch.from_numpy(ts_arr).to(device)
    pr = torch.from_numpy(pr_arr).to(device)
    mask = torch.from_numpy(mask_arr).to(device)
    target_pi = torch.from_numpy(pi_arr).to(device)
    z = torch.from_numpy(z_arr).to(device)
    wdl = torch.from_numpy(wdl_arr).to(device)
    plies_left = torch.from_numpy(pl_arr).to(device)

    return boards, fs, ts, pr, mask, target_pi, z, wdl, plies_left


def masked_entropy(p: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    ent = -(p * p.log()).masked_fill(~mask, 0.0).sum(dim=-1)
    return ent.mean()


def train_step(net, opt, batch, val_w=2.0, moves_left_w: float = 0.15,
               loss_scale: float = 1.0, do_step: bool = True):
    """Single forward+backward pass.

    Args:
        loss_scale: multiply loss before backward (for gradient accumulation,
                    pass 1/accum_steps so gradients average correctly).
        do_step:    if False, skip opt.zero_grad before and opt.step after
                    (caller manages these for accumulation).
    """
    boards, fs, ts, pr, mask, target_pi, z, wdl_target, plies_left_target = batch
    logits, wdl_logits, moves_left_pred = net.forward_policy_value(boards, fs, ts, pr, mask)

    logp = F.log_softmax(logits, dim=-1)
    pol_loss = -(target_pi * logp).masked_fill(~mask, 0.0).sum(dim=-1).mean()

    # WDL cross-entropy loss (replaces Huber on scalar z)
    wdl_logp = F.log_softmax(wdl_logits, dim=-1)  # (B, 3)
    val_loss = -(wdl_target * wdl_logp).sum(dim=-1).mean()
    ml_loss = F.smooth_l1_loss(moves_left_pred, plies_left_target)

    loss = pol_loss + val_w * val_loss + float(moves_left_w) * ml_loss

    if do_step:
        opt.zero_grad()
    (loss * loss_scale).backward()

    grad_norm = 0.0
    if do_step:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0))
        opt.step()

    with torch.no_grad():
        pred_pi = torch.softmax(logits, dim=-1)
        pred_ent = float(masked_entropy(pred_pi, mask).item())
        tgt_ent = float(masked_entropy(target_pi, mask).item())
        # Convert WDL to scalar for logging: v = P(win) - P(loss)
        wdl_probs = torch.softmax(wdl_logits, dim=-1)
        v = wdl_probs[:, 0] - wdl_probs[:, 2]  # win - loss
        v_mean = float(v.mean().item())
        z_mean = float(z.mean().item())
        ml_mean = float(moves_left_pred.mean().item())
        ml_tgt_mean = float(plies_left_target.mean().item())

        if v.numel() > 2:
            vv = v.detach().float()
            zz = z.detach().float()
            if (vv.std(unbiased=False) > 1e-6) and (zz.std(unbiased=False) > 1e-6):
                vz_corr = float(torch.corrcoef(torch.stack([vv, zz]))[0, 1].item())
            else:
                vz_corr = 0.0
        else:
            vz_corr = 0.0

        if not math.isfinite(vz_corr):
            vz_corr = 0.0

    return {
        "loss": float(loss.item()),
        "pol": float(pol_loss.item()),
        "val": float(val_loss.item()),
        "ml": float(ml_loss.item()),
        "grad_norm": grad_norm,
        "pred_ent": pred_ent,
        "tgt_ent": tgt_ent,
        "v_mean": v_mean,
        "z_mean": z_mean,
        "vz_corr": vz_corr,
        "ml_mean": ml_mean,
        "ml_tgt_mean": ml_tgt_mean,
    }
