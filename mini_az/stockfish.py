"""Stockfish helpers: engine open, eval, teacher policy, cp_to_z."""

import math
from collections import OrderedDict
from typing import Dict, Optional, Hashable

import numpy as np
import chess
import chess.engine

from .config import print  # flush-print


class SfTeacherCache:
    """Per-worker LRU cache for Stockfish teacher analysis results.

    Stores the intermediate `move → cp` dict (before softmax) keyed by
    (transposition_key, depth, multipv, cp_cap). cp_soft_scale and eps
    affect only the post-softmax transformation so they are NOT part of
    the key — we always re-softmax on hit. Engine calls cost ~50 ms;
    dict lookup + softmax is ~100 us, so a ~20% hit rate reclaims a
    meaningful share of worker wall-time, especially in openings and
    forced-sequence mid-games where transpositions are common.
    """

    def __init__(self, max_entries: int = 10000):
        self.max = int(max_entries)
        self._cache: "OrderedDict[Hashable, Dict[chess.Move, int]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(board: chess.Board, depth, multipv, cp_cap) -> Hashable:
        try:
            bk = board._transposition_key()
        except Exception:
            bk = board.fen()
        return (bk, depth, int(multipv), int(cp_cap) if cp_cap is not None else -1)

    def get(self, board, depth, multipv, cp_cap) -> Optional[Dict[chess.Move, int]]:
        k = self._make_key(board, depth, multipv, cp_cap)
        v = self._cache.get(k)
        if v is None:
            self.misses += 1
            return None
        self._cache.move_to_end(k)
        self.hits += 1
        return v

    def put(self, board, depth, multipv, cp_cap, move_cp: Dict[chess.Move, int]) -> None:
        k = self._make_key(board, depth, multipv, cp_cap)
        self._cache[k] = move_cp
        self._cache.move_to_end(k)
        while len(self._cache) > self.max:
            self._cache.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "hit_rate": (self.hits / total) if total > 0 else 0.0,
        }


def elo_diff_from_score(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10((1.0 / score) - 1.0)


def sf_eval_cp_white(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    movetime_ms: int = 15,
    depth: int | None = None,
    mate_cp: int = 10000,
) -> Optional[int]:
    try:
        limit = chess.engine.Limit(time=movetime_ms / 1000.0) if depth is None else chess.engine.Limit(depth=depth)
        info = engine.analyse(board, limit)

        score = info.get("score")
        if score is None:
            return None

        s = score.pov(chess.WHITE)
        if s.is_mate():
            m = s.mate()
            if m is None:
                return 0
            return mate_cp if m > 0 else -mate_cp

        cp = s.score(mate_score=mate_cp)
        return None if cp is None else int(cp)
    except Exception:
        return None


def cp_to_z(cp_white: int, cp_scale: float = 1500.0, cp_cap: int = 1500) -> float:
    cp = int(cp_white)
    if cp_cap is not None:
        cp = max(-int(cp_cap), min(int(cp_cap), cp))

    x = float(cp) / float(cp_scale)
    return float(np.tanh(x))


def _sf_analyse_move_cp(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    legal_moves: list,
    limit: chess.engine.Limit,
    multipv: int,
    mate_cp: int,
    cp_cap: Optional[int],
) -> Dict[chess.Move, int]:
    """Run one engine.analyse and return the {move: cp_from_turn_perspective}
    dict (before softmax). Isolated so the cache can store it."""
    infos = engine.analyse(board, limit, multipv=int(multipv))
    if isinstance(infos, dict):
        infos = [infos]

    move_cp: Dict[chess.Move, int] = {}
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        mv0 = pv[0]
        if mv0 not in legal_moves:
            continue
        score = info.get("score")
        if score is None:
            continue

        s_white = score.pov(chess.WHITE)
        if s_white.is_mate():
            m = s_white.mate()
            if m is None:
                cp_white = 0
            else:
                cp_white = mate_cp if m > 0 else -mate_cp
        else:
            cp = s_white.score(mate_score=mate_cp)
            if cp is None:
                continue
            cp_white = int(cp)

        cp_turn = cp_white if board.turn == chess.WHITE else -cp_white
        if cp_cap is not None:
            cp_turn = max(-int(cp_cap), min(int(cp_cap), int(cp_turn)))

        old = move_cp.get(mv0, None)
        if (old is None) or (cp_turn > old):
            move_cp[mv0] = cp_turn
    return move_cp


def _move_cp_to_policy(
    legal_moves: list,
    move_cp: Dict[chess.Move, int],
    cp_soft_scale: float,
    eps: float,
) -> Optional[np.ndarray]:
    """Apply softmax + eps smoothing over legal moves. Cheap (~100 us)."""
    if not move_cp:
        return None
    L = len(legal_moves)
    logits = np.full((L,), -1e9, dtype=np.float64)
    for i, mv in enumerate(legal_moves):
        if mv in move_cp:
            logits[i] = float(move_cp[mv]) / float(cp_soft_scale)

    m = np.max(logits)
    if not np.isfinite(m):
        return None
    exp = np.exp(logits - m)
    s = float(exp.sum())
    if s <= 0:
        return None
    p_top = exp / s

    e = float(np.clip(eps, 0.0, 0.5))
    p = (1.0 - e) * p_top + (e / L)
    p = p / (p.sum() + 1e-12)
    return p.astype(np.float32)


def sf_teacher_policy_legal(
    engine: "chess.engine.SimpleEngine | None",
    board: chess.Board,
    legal_moves: list,
    movetime_ms: int = 15,
    depth: int | None = None,
    multipv: int = 4,
    mate_cp: int = 10000,
    cp_cap: int = 800,
    cp_soft_scale: float = 120.0,
    eps: float = 0.01,
    cache: "SfTeacherCache | None" = None,
) -> Optional[np.ndarray]:
    """Return a length-|legal_moves| probability vector derived from Stockfish
    multipv analysis of *board*. If *cache* is provided, reuse the intermediate
    analysis on position + (depth, multipv, cp_cap) hits — saves the ~50ms
    engine call. cp_soft_scale and eps affect only the post-softmax shaping
    and are applied every call, so they can vary across callers without
    invalidating the cache.
    """
    try:
        if not legal_moves or (engine is None):
            return None

        # Only positions with a fixed depth (not movetime) are cacheable —
        # wall-clock-limited analyses give non-deterministic move_cp.
        cacheable = cache is not None and depth is not None

        if cacheable:
            cached = cache.get(board, depth, multipv, cp_cap)
            if cached is not None:
                return _move_cp_to_policy(legal_moves, cached, cp_soft_scale, eps)

        limit = (chess.engine.Limit(time=movetime_ms / 1000.0)
                 if depth is None else chess.engine.Limit(depth=depth))

        move_cp = _sf_analyse_move_cp(
            engine, board, legal_moves, limit,
            multipv=multipv, mate_cp=mate_cp, cp_cap=cp_cap,
        )

        if cacheable and move_cp:
            cache.put(board, depth, multipv, cp_cap, move_cp)

        return _move_cp_to_policy(legal_moves, move_cp, cp_soft_scale, eps)

    except Exception:
        return None


def open_stockfish_engine(
        stockfish_path: str,
        threads: int = 1,
        hash_mb: int = 16,
        elo: int | None = None,
        skill: int | None = None) -> chess.engine.SimpleEngine:
    cmd = [stockfish_path]

    try:
        engine = chess.engine.SimpleEngine.popen_uci(cmd)
    except Exception as e:
        raise RuntimeError(
            f"Cannot launch Stockfish: path={stockfish_path!r}. "
            f"Check that the binary exists and is executable. Original error: {e}"
        )

    if elo is not None:
        if ("UCI_LimitStrength" not in engine.options) or ("UCI_Elo" not in engine.options):
            raise RuntimeError(
                "This Stockfish build does not support UCI_LimitStrength/UCI_Elo — "
                "Elo-limited evaluation would be unreliable. "
                f"Available options: {list(engine.options.keys())[:40]} ..."
            )

    cfg = {"Threads": int(threads), "Hash": int(hash_mb)}

    try:
        if elo is not None and ("UCI_LimitStrength" in engine.options) and ("UCI_Elo" in engine.options):
            cfg["UCI_LimitStrength"] = True
            cfg["UCI_Elo"] = int(elo)
        elif skill is not None and ("Skill Level" in engine.options):
            cfg["Skill Level"] = int(skill)

        engine.configure(cfg)

        print(f"[SF] configured Threads={cfg.get('Threads')} Hash={cfg.get('Hash')} "
              f"LimitStrength={cfg.get('UCI_LimitStrength', None)} Elo={cfg.get('UCI_Elo', None)} "
              f"Skill={cfg.get('Skill Level', None)}")
    except Exception as e:
        raise RuntimeError(f"[SF] configure failed: {e} cfg={cfg}")

    return engine

