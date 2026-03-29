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

    def sample_batch_mixed(self, batch_size: int, recent_frac: float = 0.5, recent_window: int = 200_000) -> list[Sample]:
        size = len(self.data)
        if size == 0:
            return []

        recent_frac = float(np.clip(recent_frac, 0.0, 1.0))
        n_recent = int(round(batch_size * recent_frac))
        n_all = batch_size - n_recent

        idxs = []
        if n_recent > 0:
            idxs.append(self._sample_indices_recent(n_recent, recent_window))
        if n_all > 0:
            idxs.append(self._sample_indices_uniform(n_all))

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
    return Sample(bp, fs, ts, s.moves_pr.copy(), s.target_pi.copy(), s.z, s.wdl.copy())


def collate(samples: List[Sample], device: str):
    aug = []
    for s in samples:
        if random.random() < 0.5:
            aug.append(flip_sample_lr(s))
        else:
            aug.append(s)
    samples = aug
    B = len(samples)
    Lmax = max(s.moves_fs.shape[0] for s in samples)

    boards = torch.tensor(np.stack([s.board_planes for s in samples]), dtype=torch.float32, device=device)
    fs = torch.full((B, Lmax), 0, dtype=torch.long, device=device)
    ts = torch.full((B, Lmax), 0, dtype=torch.long, device=device)
    pr = torch.full((B, Lmax), 0, dtype=torch.long, device=device)
    mask = torch.zeros((B, Lmax), dtype=torch.bool, device=device)
    target_pi = torch.zeros((B, Lmax), dtype=torch.float32, device=device)
    z = torch.tensor([s.z for s in samples], dtype=torch.float32, device=device)
    wdl = torch.tensor(np.stack([s.wdl for s in samples]), dtype=torch.float32, device=device)  # (B, 3)

    for i, s in enumerate(samples):
        L = s.moves_fs.shape[0]
        fs[i, :L] = torch.tensor(s.moves_fs, dtype=torch.long, device=device)
        ts[i, :L] = torch.tensor(s.moves_ts, dtype=torch.long, device=device)
        pr[i, :L] = torch.tensor(s.moves_pr, dtype=torch.long, device=device)
        mask[i, :L] = True
        target_pi[i, :L] = torch.tensor(s.target_pi, dtype=torch.float32, device=device)

    return boards, fs, ts, pr, mask, target_pi, z, wdl


def masked_entropy(p: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    ent = -(p * p.log()).masked_fill(~mask, 0.0).sum(dim=-1)
    return ent.mean()


def train_step(net, opt, batch, val_w=2.0):
    boards, fs, ts, pr, mask, target_pi, z, wdl_target = batch
    logits, wdl_logits = net.forward_policy_value(boards, fs, ts, pr, mask)

    logp = F.log_softmax(logits, dim=-1)
    pol_loss = -(target_pi * logp).masked_fill(~mask, 0.0).sum(dim=-1).mean()

    # WDL cross-entropy loss (replaces Huber on scalar z)
    wdl_logp = F.log_softmax(wdl_logits, dim=-1)  # (B, 3)
    val_loss = -(wdl_target * wdl_logp).sum(dim=-1).mean()

    loss = pol_loss + val_w * val_loss

    opt.zero_grad()
    loss.backward()
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
        "grad_norm": grad_norm,
        "pred_ent": pred_ent,
        "tgt_ent": tgt_ent,
        "v_mean": v_mean,
        "z_mean": z_mean,
        "vz_corr": vz_corr,
    }
