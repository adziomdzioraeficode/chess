#!/usr/bin/env python3
"""
V3 Architecture Benchmark — find optimal worker count for Azure D96s v6.

Tests:
  1. Single-core inference latency (network forward + MCTS search)
  2. Parallel MCTS throughput at various worker counts
  3. Training step throughput at various thread counts
  4. Stockfish teacher latency (if available)

Architecture: 10×96 SE-ResNet, 45 input planes, WDL head, ~5.5M params.
Target: Azure Standard_D96s_v6 = 48 physical / 96 HT cores (AMD EPYC).
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
import torch
import chess


def _suppress_threads():
    """Pin worker to 1 thread for maximum parallelism."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# 1. Single-core latency
# ---------------------------------------------------------------------------
def bench_single_core():
    _suppress_threads()
    from mini_az.network import ChessNet
    from mini_az.mcts import mcts_search
    from mini_az.encoding import board_to_tensor, INPUT_PLANES, HISTORY_STEPS

    net = ChessNet()
    net.eval()
    params = sum(p.numel() for p in net.parameters())
    b = chess.Board()

    # Warm up
    for _ in range(3):
        mcts_search(net, b, "cpu", sims=16, history=[])

    # Single forward pass
    t = board_to_tensor(b, history=[]).unsqueeze(0)
    n_fwd = 50
    t0 = time.perf_counter()
    for _ in range(n_fwd):
        with torch.no_grad():
            net.encode_board(t)
    fwd_ms = (time.perf_counter() - t0) / n_fwd * 1000

    # MCTS search (64 sims — same as selfplay)
    n_mcts = 10
    t0 = time.perf_counter()
    for _ in range(n_mcts):
        mcts_search(net, b, "cpu", sims=64, history=[])
    mcts_ms = (time.perf_counter() - t0) / n_mcts * 1000

    # MCTS search (400 sims — eval)
    n_eval = 3
    t0 = time.perf_counter()
    for _ in range(n_eval):
        mcts_search(net, b, "cpu", sims=400, history=[])
    eval_ms = (time.perf_counter() - t0) / n_eval * 1000

    return {
        "params": params,
        "fwd_ms": fwd_ms,
        "mcts_64_ms": mcts_ms,
        "mcts_400_ms": eval_ms,
    }


# ---------------------------------------------------------------------------
# 2. Parallel MCTS throughput
# ---------------------------------------------------------------------------
def _parallel_worker(results_q, worker_id, sims, n_games):
    _suppress_threads()
    from mini_az.network import ChessNet
    from mini_az.mcts import mcts_search

    net = ChessNet()
    net.eval()
    b = chess.Board()

    # Warm up
    for _ in range(2):
        mcts_search(net, b, "cpu", sims=sims, history=[])

    t0 = time.perf_counter()
    for _ in range(n_games):
        mcts_search(net, b, "cpu", sims=sims, history=[])
    elapsed = time.perf_counter() - t0

    results_q.put((worker_id, elapsed, n_games))


def bench_parallel(worker_counts, sims=64, games_per_worker=6):
    results = {}
    for nw in worker_counts:
        q = mp.Queue()
        procs = []
        for i in range(nw):
            p = mp.Process(target=_parallel_worker, args=(q, i, sims, games_per_worker))
            p.start()
            procs.append(p)

        wall_t0 = time.perf_counter()
        for p in procs:
            p.join(timeout=300)
        wall_elapsed = time.perf_counter() - wall_t0

        worker_results = []
        while not q.empty():
            worker_results.append(q.get_nowait())

        if not worker_results:
            results[nw] = None
            continue

        total_games = sum(r[2] for r in worker_results)
        worker_times = [r[1] for r in worker_results]
        avg_time = sum(worker_times) / len(worker_times)
        max_time = max(worker_times)
        min_time = min(worker_times)

        # Throughput = total searches completed / wall time
        throughput = total_games / wall_elapsed
        # Per-worker avg ms/search
        avg_ms = avg_time / games_per_worker * 1000
        max_ms = max_time / games_per_worker * 1000

        results[nw] = {
            "workers_ok": len(worker_results),
            "total_games": total_games,
            "wall_s": wall_elapsed,
            "throughput": throughput,
            "avg_ms": avg_ms,
            "max_ms": max_ms,
            "min_ms": min_time / games_per_worker * 1000,
            "slowdown": max_ms / (avg_ms + 1e-9),
        }

    return results


# ---------------------------------------------------------------------------
# 3. Training throughput
# ---------------------------------------------------------------------------
def bench_training(thread_counts, batch_size=512, steps=20):
    from mini_az.network import ChessNet
    from mini_az.encoding import board_to_tensor, legal_moves_canonical
    from mini_az.training import Sample, ReplayBuffer, collate, train_step

    torch.set_grad_enabled(True)

    b = chess.Board()
    t = board_to_tensor(b, history=[])
    legals = legal_moves_canonical(b)
    fs = np.array([m[0] for m in legals], dtype=np.int64)
    ts = np.array([m[1] for m in legals], dtype=np.int64)
    pr = np.array([m[2] for m in legals], dtype=np.int64)
    pi = np.ones(len(legals), dtype=np.float32) / len(legals)
    wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    buf = ReplayBuffer(10000)
    samples = [Sample(t.numpy(), fs, ts, pr, pi, 0.0, wdl) for _ in range(2000)]
    buf.add_game(samples)

    results = {}
    for nt in thread_counts:
        torch.set_num_threads(nt)
        net = ChessNet()
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

        # Warm up
        for _ in range(3):
            batch = buf.sample_batch(batch_size)
            coll = collate(batch, "cpu")
            train_step(net, opt, coll)

        t0 = time.perf_counter()
        for _ in range(steps):
            batch = buf.sample_batch(batch_size)
            coll = collate(batch, "cpu")
            train_step(net, opt, coll)
        elapsed = time.perf_counter() - t0

        steps_per_s = steps / elapsed
        ms_per_step = elapsed / steps * 1000
        results[nt] = {
            "steps_per_s": steps_per_s,
            "ms_per_step": ms_per_step,
        }

    return results


# ---------------------------------------------------------------------------
# 4. SF teacher latency
# ---------------------------------------------------------------------------
def bench_sf_teacher():
    try:
        from mini_az.stockfish import open_stockfish_engine, sf_teacher_policy_legal
        sf = open_stockfish_engine(
            stockfish_path="stockfish", threads=1, hash_mb=16,
            elo=2000, skill=None,
        )
    except Exception as e:
        return {"error": str(e)}

    b = chess.Board()
    legals = list(b.legal_moves)

    # depth-14 (as in train_fresh.sh)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        sf_teacher_policy_legal(
            sf, b, legals,
            movetime_ms=0, depth=14, multipv=5,
            mate_cp=10000, cp_cap=600, cp_soft_scale=150.0, eps=0.02,
        )
    elapsed_d14 = (time.perf_counter() - t0) / n * 1000

    # depth-8 (lighter option)
    t0 = time.perf_counter()
    for _ in range(n):
        sf_teacher_policy_legal(
            sf, b, legals,
            movetime_ms=0, depth=8, multipv=5,
            mate_cp=10000, cp_cap=600, cp_soft_scale=150.0, eps=0.02,
        )
    elapsed_d8 = (time.perf_counter() - t0) / n * 1000

    sf.quit()
    return {"depth_14_ms": elapsed_d14, "depth_8_ms": elapsed_d8}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mp.set_start_method("spawn", force=True)

    ncpu = os.cpu_count() or 4
    print(f"CPU cores detected: {ncpu}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: num_threads={torch.get_num_threads()}, "
          f"num_interop_threads={torch.get_num_interop_threads()}")
    print()

    # --- 1. Single-core ---
    print("=" * 60)
    print("1. SINGLE-CORE LATENCY (v3: 10×96, 45 planes, ~5.5M params)")
    print("=" * 60)
    sc = bench_single_core()
    print(f"  Parameters:       {sc['params']:,}")
    print(f"  Forward pass:     {sc['fwd_ms']:.1f} ms")
    print(f"  MCTS  64 sims:    {sc['mcts_64_ms']:.0f} ms")
    print(f"  MCTS 400 sims:    {sc['mcts_400_ms']:.0f} ms")
    print()

    # --- 2. Parallel MCTS ---
    print("=" * 60)
    print("2. PARALLEL MCTS THROUGHPUT (64 sims/search)")
    print("=" * 60)

    # Scale worker counts to machine
    if ncpu >= 90:
        # Azure D96s v6 range
        worker_counts = [40, 48, 56, 64, 72, 80, 84, 88, 92]
    elif ncpu >= 16:
        worker_counts = [4, 8, 12, ncpu // 2, ncpu - 4, ncpu - 2, ncpu]
    else:
        worker_counts = [1, 2, 4, max(4, ncpu - 1), ncpu]

    par = bench_parallel(worker_counts, sims=64, games_per_worker=6)
    print(f"  {'Workers':>7} {'Throughput':>12} {'Avg ms':>8} {'Max ms':>8} "
          f"{'Slowdown':>9} {'Wall s':>7}")
    best_nw = 0
    best_tp = 0
    for nw in worker_counts:
        r = par.get(nw)
        if r is None:
            print(f"  {nw:>7} {'FAILED':>12}")
            continue
        marker = ""
        if r["throughput"] > best_tp:
            best_tp = r["throughput"]
            best_nw = nw
        print(f"  {nw:>7} {r['throughput']:>10.1f}/s {r['avg_ms']:>7.0f} "
              f"{r['max_ms']:>7.0f} {r['slowdown']:>8.2f}x {r['wall_s']:>6.1f}")
    print(f"\n  >>> Best parallel throughput: {best_nw} workers = {best_tp:.1f} searches/s")
    print()

    # --- 3. Training throughput ---
    print("=" * 60)
    print("3. TRAINING THROUGHPUT (batch=512, 10×96 net)")
    print("=" * 60)

    if ncpu >= 90:
        train_threads = [8, 16, 24, 32, 48]
    elif ncpu >= 16:
        train_threads = [4, 8, ncpu // 2, ncpu]
    else:
        train_threads = [2, 4, ncpu]

    tr = bench_training(train_threads, batch_size=512, steps=20)
    print(f"  {'Threads':>7} {'Steps/s':>10} {'ms/step':>10}")
    best_nt = 0
    best_sps = 0
    for nt in train_threads:
        r = tr.get(nt)
        if r is None:
            continue
        if r["steps_per_s"] > best_sps:
            best_sps = r["steps_per_s"]
            best_nt = nt
        print(f"  {nt:>7} {r['steps_per_s']:>9.1f} {r['ms_per_step']:>9.0f}")
    print(f"\n  >>> Best training: {best_nt} threads = {best_sps:.1f} steps/s")
    print()

    # --- 4. Stockfish teacher ---
    print("=" * 60)
    print("4. STOCKFISH TEACHER LATENCY")
    print("=" * 60)
    sf = bench_sf_teacher()
    if "error" in sf:
        print(f"  Stockfish not available: {sf['error']}")
    else:
        print(f"  Depth 14, multipv=5: {sf['depth_14_ms']:.1f} ms/call")
        print(f"  Depth  8, multipv=5: {sf['depth_8_ms']:.1f} ms/call")
    print()

    # --- Summary and recommendations ---
    print("=" * 60)
    print("RECOMMENDATIONS FOR train_fresh_v3.sh")
    print("=" * 60)

    # Estimate: each selfplay game ≈ 60 MCTS searches × mcts_64_ms (from root)
    # But with SF teacher overhead (50% prob), add ~30 SF calls
    avg_game_plies = 70  # typical game length
    avg_searches_per_game = avg_game_plies  # 1 search per ply
    single_game_time_s = avg_searches_per_game * sc["mcts_64_ms"] / 1000

    if "error" not in sf:
        sf_overhead_s = avg_searches_per_game * 0.5 * sf["depth_14_ms"] / 1000  # 50% teacher prob
        single_game_time_s += sf_overhead_s

    games_per_iter = 120
    if best_tp > 0:
        # Wall time for selfplay = games_per_iter * searches_per_game / throughput
        sp_wall_s = games_per_iter * avg_searches_per_game / best_tp
    else:
        sp_wall_s = float("inf")

    train_steps = 100
    if best_sps > 0:
        train_wall_s = train_steps / best_sps
    else:
        train_wall_s = float("inf")

    iter_total_s = sp_wall_s + train_wall_s
    print(f"  Single game (serial):        ~{single_game_time_s:.1f}s")
    print(f"  Selfplay ({games_per_iter} games, {best_nw}w): ~{sp_wall_s:.0f}s")
    print(f"  Training ({train_steps} steps, {best_nt}t):   ~{train_wall_s:.0f}s")
    print(f"  Estimated iter time:          ~{iter_total_s:.0f}s")
    print(f"  Iters in 8h:                  ~{8*3600/iter_total_s:.0f}")
    print()

    rec_workers = best_nw
    print(f"  Recommended --workers:         {rec_workers}")
    print(f"  Recommended train threads:     {best_nt}")
    print(f"  Recommended --games_per_iter:  {games_per_iter}")
    print(f"  Recommended --steps_per_iter:  {train_steps}")


if __name__ == "__main__":
    main()
