"""Tests for the SfTeacherCache (Etap 4.1).

Uses a fake engine so tests do not depend on a Stockfish binary being
installed in the test environment.
"""

import numpy as np
import chess
import chess.engine

from mini_az.stockfish import SfTeacherCache, sf_teacher_policy_legal


class FakeEngine:
    """Minimal stand-in for chess.engine.SimpleEngine.analyse.

    Returns an info-list that marks the first legal move as winning (cp=+100)
    and the rest as slightly worse (cp=-10). Counts how many times analyse
    was invoked so tests can assert cache hits/misses.
    """

    def __init__(self):
        self.calls = 0

    def analyse(self, board, limit, multipv=1):
        self.calls += 1
        legal = list(board.legal_moves)
        infos = []
        for i, mv in enumerate(legal[: int(multipv)]):
            cp = 100 if i == 0 else -10
            infos.append({
                "pv": [mv],
                "score": chess.engine.PovScore(
                    chess.engine.Cp(cp if board.turn == chess.WHITE else -cp),
                    chess.WHITE,
                ),
            })
        return infos


def test_cache_hit_returns_equal_policy():
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine()
    b = chess.Board()
    legal = list(b.legal_moves)

    p1 = sf_teacher_policy_legal(
        eng, b, legal, depth=4, multipv=4, cp_cap=600, cp_soft_scale=120.0,
        eps=0.01, cache=cache,
    )
    assert p1 is not None
    assert eng.calls == 1
    assert cache.stats()["misses"] == 1

    p2 = sf_teacher_policy_legal(
        eng, b, legal, depth=4, multipv=4, cp_cap=600, cp_soft_scale=120.0,
        eps=0.01, cache=cache,
    )
    assert eng.calls == 1, "cached hit must not touch the engine"
    assert cache.stats()["hits"] == 1
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_cache_eps_and_cp_soft_scale_not_in_key():
    """Changing eps / cp_soft_scale must still be a cache hit and must
    produce a different policy (post-softmax shaping applied every call)."""
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine()
    b = chess.Board()
    legal = list(b.legal_moves)

    p1 = sf_teacher_policy_legal(
        eng, b, legal, depth=4, multipv=4, cp_cap=600, cp_soft_scale=120.0,
        eps=0.01, cache=cache,
    )
    p2 = sf_teacher_policy_legal(
        eng, b, legal, depth=4, multipv=4, cp_cap=600, cp_soft_scale=240.0,
        eps=0.20, cache=cache,
    )
    assert eng.calls == 1, "eps / cp_soft_scale must not invalidate the cache"
    assert cache.stats()["hits"] == 1
    assert not np.allclose(p1, p2), "post-softmax shaping differs → policies differ"


def test_cache_depth_is_part_of_key():
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine()
    b = chess.Board()
    legal = list(b.legal_moves)

    sf_teacher_policy_legal(eng, b, legal, depth=4, multipv=4, cp_cap=600,
                            cache=cache)
    sf_teacher_policy_legal(eng, b, legal, depth=8, multipv=4, cp_cap=600,
                            cache=cache)
    assert eng.calls == 2, "different depth → different key → two engine calls"
    assert cache.stats()["misses"] == 2


def test_cache_multipv_is_part_of_key():
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine()
    b = chess.Board()
    legal = list(b.legal_moves)

    sf_teacher_policy_legal(eng, b, legal, depth=4, multipv=2, cp_cap=600,
                            cache=cache)
    sf_teacher_policy_legal(eng, b, legal, depth=4, multipv=5, cp_cap=600,
                            cache=cache)
    assert eng.calls == 2


def test_cache_movetime_path_uncached():
    """Movetime analyses are wall-clock-limited and non-deterministic; they
    must NOT be cached even when a cache is passed."""
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine()
    b = chess.Board()
    legal = list(b.legal_moves)

    sf_teacher_policy_legal(eng, b, legal, movetime_ms=5, depth=None,
                            multipv=4, cp_cap=600, cache=cache)
    sf_teacher_policy_legal(eng, b, legal, movetime_ms=5, depth=None,
                            multipv=4, cp_cap=600, cache=cache)
    assert eng.calls == 2
    assert cache.stats()["size"] == 0
    assert cache.stats()["hits"] == 0


def test_cache_lru_eviction():
    cache = SfTeacherCache(max_entries=2)
    eng = FakeEngine()

    b1 = chess.Board()
    b2 = b1.copy(); b2.push_san("e4")
    b3 = b1.copy(); b3.push_san("d4")

    for b in (b1, b2, b3):
        sf_teacher_policy_legal(
            eng, b, list(b.legal_moves),
            depth=4, multipv=4, cp_cap=600, cache=cache,
        )
    assert cache.stats()["size"] == 2, "oldest entry must have been evicted"
    # Re-querying b1 should now be a miss (evicted).
    sf_teacher_policy_legal(eng, b1, list(b1.legal_moves),
                            depth=4, multipv=4, cp_cap=600, cache=cache)
    assert eng.calls == 4


def test_cache_disabled_when_cache_is_none():
    eng = FakeEngine()
    b = chess.Board()
    legal = list(b.legal_moves)
    sf_teacher_policy_legal(eng, b, legal, depth=4, multipv=4, cp_cap=600,
                            cache=None)
    sf_teacher_policy_legal(eng, b, legal, depth=4, multipv=4, cp_cap=600,
                            cache=None)
    assert eng.calls == 2
