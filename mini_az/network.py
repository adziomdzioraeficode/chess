"""Neural network: residual tower + policy/value heads."""

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .encoding import INPUT_PLANES, board_to_tensor, legal_moves_canonical
from .config import Policy, PV


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns channel-wise attention weights."""
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        mid = max(channels // ratio, 1)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.fc2 = nn.Linear(mid, 2 * channels, bias=False)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        z = x.mean(dim=(2, 3))                       # (B, C)
        z = F.relu(self.fc1(z))                       # (B, mid)
        z = self.fc2(z)                               # (B, 2C)
        gamma, beta = z[:, :C], z[:, C:]              # (B, C) each
        gamma = torch.sigmoid(gamma).view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return x * gamma + beta


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, gn_groups: int, se_ratio: int = 4):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.g1 = nn.GroupNorm(gn_groups, channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.g2 = nn.GroupNorm(gn_groups, channels)
        self.se = SEBlock(channels, ratio=se_ratio)

    def forward(self, x):
        h = F.relu(self.g1(self.c1(x)))
        h = self.g2(self.c2(h))
        h = self.se(h)
        return F.relu(x + h)


class ChessNet(nn.Module):
    def __init__(self, channels=96, num_blocks=10, emb_dim=32, board_dim=512, gn_groups=8, dropout: float = 0.10, se_ratio: int = 4):
        super().__init__()
        self.dropout = float(dropout)
        self.board_fc = nn.Linear(channels * 8 * 8, board_dim)

        def _good_gn_groups(c: int, want: int) -> int:
            g = min(want, c)
            while g > 1 and (c % g) != 0:
                g -= 1
            return g

        gn_groups = _good_gn_groups(channels, gn_groups)

        self.stem = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.stem_gn = nn.GroupNorm(gn_groups, channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels, gn_groups, se_ratio=se_ratio) for _ in range(num_blocks)])
        self.from_emb = nn.Embedding(64, emb_dim)
        self.to_emb   = nn.Embedding(64, emb_dim)
        self.promo_emb = nn.Embedding(5, emb_dim)

        self.move_fc1 = nn.Linear(board_dim + 3 * emb_dim, board_dim)
        self.move_fc2 = nn.Linear(board_dim, 1)

        # WDL value head: 3 outputs (win, draw, loss) instead of single tanh
        self.val_fc1 = nn.Linear(board_dim, board_dim)
        self.val_fc2 = nn.Linear(board_dim, 3)
        # Moves-left head: regularizes search toward faster conversion / less shuffling.
        self.ml_fc1 = nn.Linear(board_dim, board_dim // 2)
        self.ml_fc2 = nn.Linear(board_dim // 2, 1)

    def encode_board(self, x):
        x = F.relu(self.stem_gn(self.stem(x)))
        for blk in self.blocks:
            x = blk(x)
        x = x.flatten(1)
        x = F.relu(self.board_fc(x))
        return x

    def forward_policy_value(
        self,
        board_tensor: torch.Tensor,
        from_sqs: torch.Tensor,
        to_sqs: torch.Tensor,
        promos: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Returns (policy_logits, wdl_logits, moves_left_pred).

        wdl_logits: (B, 3) raw logits for [win, draw, loss].
        Use softmax to get probabilities, then v = P(win) - P(loss) for scalar value.
        moves_left_pred: (B,) predicted remaining plies to game end.
        """
        B, L = from_sqs.shape
        bemb = self.encode_board(board_tensor)

        me = torch.cat([self.from_emb(from_sqs), self.to_emb(to_sqs), self.promo_emb(promos)], dim=-1)
        bemb_exp = bemb.unsqueeze(1).expand(-1, L, -1)
        h = torch.cat([bemb_exp, me], dim=-1)
        h = F.relu(self.move_fc1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.move_fc2(h).squeeze(-1)

        logits = logits.masked_fill(~mask, float("-inf"))
        vh = F.relu(self.val_fc1(bemb))
        vh = F.dropout(vh, p=self.dropout, training=self.training)
        wdl_logits = self.val_fc2(vh)  # (B, 3)
        mh = F.relu(self.ml_fc1(bemb))
        mh = F.dropout(mh, p=self.dropout, training=self.training)
        moves_left = F.softplus(self.ml_fc2(mh)).squeeze(-1)
        return logits, wdl_logits, moves_left

    @torch.no_grad()
    def policy_value_single(self, board: chess.Board, device: str,
                            history: list = None) -> PV:
        return self.policy_value_batch([board], device, histories=[history])[0]

    @torch.no_grad()
    def policy_value_batch(self, boards: list, device: str,
                            histories: list = None) -> list:
        """Batched inference: N boards → N (priors_dict, v_scalar).

        Legal-move lists vary per board, so we pad to Lmax with a False mask.
        One forward pass amortizes kernel/Python overhead across the batch —
        the main win vs calling policy_value_single in a loop.
        """
        if not boards:
            return []
        if histories is None:
            histories = [None] * len(boards)

        per_board_moves = [legal_moves_canonical(b) for b in boards]
        B = len(boards)
        Lmax = max((len(m) for m in per_board_moves), default=1) or 1

        bt = torch.stack(
            [board_to_tensor(b, history=h) for b, h in zip(boards, histories)]
        ).to(device)
        fs = torch.zeros((B, Lmax), dtype=torch.long, device=device)
        ts = torch.zeros((B, Lmax), dtype=torch.long, device=device)
        pr = torch.zeros((B, Lmax), dtype=torch.long, device=device)
        mask = torch.zeros((B, Lmax), dtype=torch.bool, device=device)
        for i, mvs in enumerate(per_board_moves):
            L = len(mvs)
            if L == 0:
                continue
            fs[i, :L] = torch.tensor([m[0] for m in mvs], dtype=torch.long, device=device)
            ts[i, :L] = torch.tensor([m[1] for m in mvs], dtype=torch.long, device=device)
            pr[i, :L] = torch.tensor([m[2] for m in mvs], dtype=torch.long, device=device)
            mask[i, :L] = True

        # bfloat16 autocast: AVX-512-BF16 on EPYC 9004 gives ~2x over fp32 on
        # matmul-heavy conv/FC workloads. Softmax runs in fp32 for stability.
        autocast_device = "cuda" if device.startswith("cuda") else "cpu"
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            logits, wdl_logits, _ = self.forward_policy_value(bt, fs, ts, pr, mask)
        logits_f = logits.float()
        wdl_f = wdl_logits.float()

        results = []
        for i, mvs in enumerate(per_board_moves):
            if not mvs:
                results.append(({}, 0.0))
                continue
            probs_list = torch.softmax(logits_f[i, : len(mvs)], dim=0).tolist()
            priors = {mvs[j][3]: probs_list[j] for j in range(len(mvs))}
            wdl_list = torch.softmax(wdl_f[i], dim=0).tolist()
            v = float(wdl_list[0] - wdl_list[2])  # P(win) - P(loss)
            results.append((priors, v))
        return results
