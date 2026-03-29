"""Board encoding, move encoding, and CSV logging."""

import os
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import chess
import torch

PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

HISTORY_STEPS = 2  # number of previous positions to include
PLANES_PER_POS = 12  # 6 my pieces + 6 opponent pieces
AUX_PLANES = 4 + 1 + 4  # castling(4) + ep(1) + halfmove/fullmove/check/repetition
INPUT_PLANES = PLANES_PER_POS * (1 + HISTORY_STEPS) + AUX_PLANES  # 12*3 + 9 = 45

# Auxiliary plane offsets (after all piece planes)
_AUX_BASE = PLANES_PER_POS * (1 + HISTORY_STEPS)  # 36
PL_MY_OO   = _AUX_BASE + 0  # 36
PL_MY_OOO  = _AUX_BASE + 1  # 37
PL_OPP_OO  = _AUX_BASE + 2  # 38
PL_OPP_OOO = _AUX_BASE + 3  # 39
PL_EP      = _AUX_BASE + 4  # 40
PL_HALFMOVE = _AUX_BASE + 5  # 41
PL_FULLMOVE = _AUX_BASE + 6  # 42
PL_CHECK    = _AUX_BASE + 7  # 43
PL_REPETITION = _AUX_BASE + 8  # 44 — how many times position appeared (0/0.5/1.0)

PROMO_MAP = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
INV_PROMO_MAP = {v: k for k, v in PROMO_MAP.items()}


def append_eval_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return

    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        old_header = next(r, None)

    if not old_header:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return

    new_keys = list(row.keys())
    if all(k in old_header for k in new_keys) and all(k in new_keys for k in old_header):
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_header)
            w.writerow(row)
        return

    union = old_header + [k for k in new_keys if k not in old_header]

    with open(path, "r", newline="") as f:
        dr = csv.DictReader(f)
        old_rows = list(dr)

    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=union)
        w.writeheader()
        for rr in old_rows:
            w.writerow(rr)
        w.writerow(row)

    os.replace(tmp, path)


def _mirror_square(sq: int) -> int:
    return chess.square_mirror(sq)


def _encode_pieces(planes: np.ndarray, board: chess.Board, turn: bool, plane_offset: int = 0):
    """Write 12 piece planes for *board* into *planes* starting at *plane_offset*.

    Perspective is always from side *turn*: "my" pieces use planes 0-5,
    "opponent" pieces use planes 6-11 (relative to plane_offset).
    Board is vertically mirrored when turn == BLACK.
    """
    for sq, piece in board.piece_map().items():
        ptype = piece.piece_type
        color = piece.color

        if turn == chess.WHITE:
            csq = sq
            my = (color == chess.WHITE)
        else:
            csq = _mirror_square(sq)
            my = (color == chess.BLACK)

        r = chess.square_rank(csq)
        f = chess.square_file(csq)

        base = PIECE_PLANES[ptype]
        idx = plane_offset + (base if my else base + 6)
        planes[idx, r, f] = 1.0


def board_to_tensor(board: chess.Board,
                    history: Optional[list] = None) -> torch.Tensor:
    """Encode the current board + up to HISTORY_STEPS previous positions.

    Parameters
    ----------
    board : chess.Board
        The current position (side-to-move perspective).
    history : list[chess.Board] | None
        Previous board states, most recent first. Up to HISTORY_STEPS entries
        are used. If None or shorter than HISTORY_STEPS, missing history slots
        are filled with zeros.
    """
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    turn = board.turn

    # --- Current position: planes 0..11 ---
    _encode_pieces(planes, board, turn, plane_offset=0)

    # --- History positions: planes 12..35 (HISTORY_STEPS * 12) ---
    if history is None:
        history = []
    for step in range(HISTORY_STEPS):
        offset = PLANES_PER_POS * (1 + step)  # 12, 24
        if step < len(history):
            _encode_pieces(planes, history[step], turn, plane_offset=offset)
        # else: leave zeros (no history available)

    # --- Auxiliary planes ---
    myc = turn
    opp = (not turn)

    if board.has_kingside_castling_rights(myc):
        planes[PL_MY_OO, :, :] = 1.0
    if board.has_queenside_castling_rights(myc):
        planes[PL_MY_OOO, :, :] = 1.0
    if board.has_kingside_castling_rights(opp):
        planes[PL_OPP_OO, :, :] = 1.0
    if board.has_queenside_castling_rights(opp):
        planes[PL_OPP_OOO, :, :] = 1.0

    if board.ep_square is not None:
        epsq = board.ep_square
        if board.turn == chess.BLACK:
            epsq = _mirror_square(epsq)
        r = chess.square_rank(epsq)
        f = chess.square_file(epsq)
        planes[PL_EP, r, f] = 1.0

    # Halfmove clock scaled to [0,1] (50-move rule = 100 halfmove_clock)
    planes[PL_HALFMOVE, :, :] = min(board.halfmove_clock / 50.0, 1.0)
    # Fullmove counter scaled to [0,1] (game phase awareness)
    planes[PL_FULLMOVE, :, :] = min(board.fullmove_number / 200.0, 1.0)
    # Is the side to move in check?
    if board.is_check():
        planes[PL_CHECK, :, :] = 1.0

    # Repetition count: 0=new, 0.5=seen once before (2 total), 1.0=twofold+ (3+ total)
    if board.is_repetition(3):
        planes[PL_REPETITION, :, :] = 1.0
    elif board.is_repetition(2):
        planes[PL_REPETITION, :, :] = 0.5

    return torch.from_numpy(planes)


def encode_move_canonical(board: chess.Board, move: chess.Move) -> Tuple[int, int, int]:
    if board.turn == chess.WHITE:
        fs, ts = move.from_square, move.to_square
    else:
        fs, ts = _mirror_square(move.from_square), _mirror_square(move.to_square)
    promo_idx = PROMO_MAP.get(move.promotion, 0)
    return fs, ts, promo_idx


def decode_move_from_canonical(board: chess.Board, fs: int, ts: int, promo_idx: int) -> chess.Move:
    promo = INV_PROMO_MAP.get(promo_idx, None)
    if board.turn == chess.WHITE:
        rfs, rts = fs, ts
    else:
        rfs, rts = _mirror_square(fs), _mirror_square(ts)
    return chess.Move(rfs, rts, promotion=promo)


def legal_moves_canonical(board: chess.Board) -> List[Tuple[int, int, int, chess.Move]]:
    out = []
    for mv in board.legal_moves:
        fs, ts, p = encode_move_canonical(board, mv)
        out.append((fs, ts, p, mv))
    return out
