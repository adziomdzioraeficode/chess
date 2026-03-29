"""Shared constants, types, opening book, and utility helpers."""

import os
import random
import sys
import time as _time
import builtins as _builtins
from typing import Dict, Tuple

import chess
import numpy as np

# --- Always-flush timestamped logging ---
_print = _builtins.print
_T0 = _time.monotonic()

def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    elapsed = _time.monotonic() - _T0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    prefix = f"[{h:02d}:{m:02d}:{s:02d}]"
    return _print(prefix, *args, **kwargs)

# Type aliases
Policy = Dict[chess.Move, float]
PV = Tuple[Policy, float]

# Threading env defaults (set before torch import)
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

# Game constants
NO_PROGRESS_HALFMOVE = 80  # 40 moves without capture/pawn push

# Mini opening book (UCI)
OPENING_BOOK_UCI: list[list[str]] = [
    ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6"],
    ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6"],
    ["e2e4","e7e5","g1f3","b8c6","d2d4","e5d4","f3d4","f8c5"],
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6"],
    ["e2e4","e7e6","d2d4","d7d5","b1c3","f8b4","e4e5","c7c5"],
    ["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4","b8d7"],
    ["e2e4","d7d6","d2d4","g8f6","b1c3","g7g6","g1f3","f8g7"],
    ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","c1g5","f8e7"],
    ["d2d4","d7d5","c2c4","c7c6","g1f3","g8f6","b1c3","d5c4"],
    ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6"],
    ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","e2e3","e8g8"],
    ["c2c4","e7e5","b1c3","g8f6","g2g3","d7d5","c4d5","f6d5"],
    ["d2d4","d7d5","g1f3","g8f6","c1f4","e7e6","e2e3","c7c5"],
]


def apply_random_opening(board: chess.Board, max_book_plies: int) -> int:
    if max_book_plies <= 0:
        return 0
    line = random.choice(OPENING_BOOK_UCI)
    k = random.randint(0, min(max_book_plies, len(line)))
    played = 0
    for uci in line[:k]:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            break
        board.push(mv)
        played += 1
    return played


def material_score(board: chess.Board) -> int:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    s = 0
    for p, v in values.items():
        s += v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
    return s
