"""Tests for the SfTeacherPrefetcher (Etap 4.2).

Uses a fake engine so tests do not depend on a Stockfish binary being
installed in the test environment.
"""

import time

import chess
import chess.engine

from mini_az.stockfish import (
    SfTeacherCache, SfTeacherPrefetcher, sf_teacher_policy_legal,
)


class FakeEngine:
    """Minimal stand-in for chess.engine.SimpleEngine.analyse with an
    optional artificial delay so we can test serialisation semantics."""

    def __init__(self, delay: float = 0.0):
        self.calls = 0
        self.delay = delay

    def analyse(self, board, limit, multipv=1):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
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


def test_prefetcher_fills_cache_in_background():
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine(delay=0.01)
    pf = SfTeacherPrefetcher(eng, cache, max_queue=4)
    try:
        b = chess.Board()
        assert pf.submit(b, list(b.legal_moves), depth=4, multipv=4, cp_cap=600)

        deadline = time.time() + 2.0
        while time.time() < deadline and cache.stats()["size"] == 0:
            time.sleep(0.01)
        assert cache.stats()["size"] == 1
        assert pf.stats()["done"] >= 1

        # The main-thread synchronous call after a prefetch must hit the cache.
        sf_teacher_policy_legal(eng, b, list(b.legal_moves),
                                depth=4, multipv=4, cp_cap=600, cache=cache)
        calls_after = eng.calls
        sf_teacher_policy_legal(eng, b, list(b.legal_moves),
                                depth=4, multipv=4, cp_cap=600, cache=cache)
        assert eng.calls == calls_after, "second sync call should be a cache hit"
    finally:
        pf.close(timeout=1.0)


def test_prefetcher_skips_when_already_cached():
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine()
    b = chess.Board()
    # Warm the cache first.
    sf_teacher_policy_legal(eng, b, list(b.legal_moves),
                            depth=4, multipv=4, cp_cap=600, cache=cache)
    pf = SfTeacherPrefetcher(eng, cache, max_queue=4)
    try:
        submitted = pf.submit(b, list(b.legal_moves),
                              depth=4, multipv=4, cp_cap=600)
        assert not submitted, "should refuse to queue when already cached"
        assert pf.stats()["skipped_hit"] >= 1
    finally:
        pf.close(timeout=1.0)


def test_prefetcher_ignores_movetime_path():
    """Prefetch only makes sense for deterministic depth-limited analyses."""
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine()
    pf = SfTeacherPrefetcher(eng, cache, max_queue=4)
    try:
        b = chess.Board()
        assert not pf.submit(b, list(b.legal_moves), depth=None, multipv=4)
    finally:
        pf.close(timeout=1.0)


def test_prefetcher_lock_serialises_main_thread():
    """When the prefetcher holds the engine lock, a concurrent lock-aware
    main-thread call must wait — proving the two paths share one engine
    safely (the engine is not thread-safe)."""
    cache = SfTeacherCache(max_entries=32)
    eng = FakeEngine(delay=0.20)
    pf = SfTeacherPrefetcher(eng, cache, max_queue=4)
    try:
        b1 = chess.Board()
        b2 = b1.copy(); b2.push_san("e4")
        pf.submit(b1, list(b1.legal_moves), depth=4, multipv=4, cp_cap=600)

        # Give the background thread a beat to grab the lock.
        time.sleep(0.02)

        t0 = time.time()
        sf_teacher_policy_legal(
            eng, b2, list(b2.legal_moves),
            depth=4, multipv=4, cp_cap=600,
            cache=cache, lock=pf.lock,
        )
        elapsed = time.time() - t0
        # Main had to queue behind the 0.20s prefetch → must take ≥ its own
        # delay plus most of the prefetch delay.
        assert elapsed >= 0.15, f"expected serialisation, got {elapsed:.3f}s"
    finally:
        pf.close(timeout=2.0)


def test_prefetcher_drops_when_queue_full():
    """Bounded queue must drop silently (best-effort) rather than block."""
    cache = SfTeacherCache(max_entries=32)
    # Give the engine a big delay so one in-flight job fills the queue and
    # subsequent submits hit the bound.
    eng = FakeEngine(delay=0.5)
    pf = SfTeacherPrefetcher(eng, cache, max_queue=1)
    try:
        b = chess.Board()
        first = pf.submit(b, list(b.legal_moves), depth=4, multipv=4, cp_cap=600)
        assert first
        # Different positions to avoid the "already cached / in-flight" short
        # circuit; submit several more than the queue can hold.
        b2 = b.copy(); b2.push_san("e4")
        b3 = b.copy(); b3.push_san("d4")
        # Let the worker thread pick up `first` so the queue has room,
        # then flood with two more (only one will fit).
        time.sleep(0.01)
        pf.submit(b2, list(b2.legal_moves), depth=4, multipv=4, cp_cap=600)
        pf.submit(b3, list(b3.legal_moves), depth=4, multipv=4, cp_cap=600)
        assert pf.stats()["dropped"] >= 1
    finally:
        pf.close(timeout=2.0)
