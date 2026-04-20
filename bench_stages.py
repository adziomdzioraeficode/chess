#!/usr/bin/env python3
"""
Stage-by-stage benchmark — measures each optimisation layer on the current machine.

Stages tested:
  Baseline:  fp32 inference, leaf_batch_size=1 (pre-Etap 1)
  Etap 1:    bf16 autocast (if avx512_bf16/amx_bf16 available)
  Etap 2:    leaf-batch K=2,4,8,16 (with best dtype from Etap 1)
  Combined:  parallel workers with best (dtype x leaf_batch) from above

Also sweeps:
  - Worker counts for selfplay throughput
  - Training thread counts
  - Batch sizes for training

Outputs a summary with recommended train_fresh_v3.sh parameters.
"""

import os
import sys
import time
import multiprocessing as mp
import subprocess

import numpy as np
import torch
import chess


def _has_bf16_hw() -> bool:
    """Check CPU flags for native bf16 support."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        return "avx512_bf16" in flags or "amx_bf16" in flags
    except Exception:
        return False


_interop_set = False

def _suppress_threads():
    global _interop_set
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if not _interop_set:
        torch.set_num_interop_threads(1)
        _interop_set = True
    torch.set_grad_enabled(False)


# ─── Etap 1: single-core bf16 vs fp32 ─────────────────────────────────────
def bench_dtype(use_bf16: bool, sims: int = 64, repeats: int = 8):
    """Single-core MCTS latency with/without bf16 autocast."""
    _suppress_threads()
    from mini_az.network import ChessNet
    from mini_az.mcts import mcts_search

    net = ChessNet()
    net.eval()
    net.use_bf16_inference = use_bf16

    b = chess.Board()
    # warm up
    for _ in range(3):
        mcts_search(net, b, "cpu", sims=sims, history=[])

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mcts_search(net, b, "cpu", sims=sims, history=[])
        times.append(time.perf_counter() - t0)

    return {
        "bf16": use_bf16,
        "sims": sims,
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }


# ─── Etap 2: leaf-batch sweep (single-core) ───────────────────────────────
def bench_leaf_batch(leaf_batch_sizes, use_bf16: bool, sims: int = 200, repeats: int = 6):
    """Single-core MCTS with varying leaf_batch_size."""
    _suppress_threads()
    from mini_az.network import ChessNet
    from mini_az.mcts import mcts_search

    net = ChessNet()
    net.eval()
    net.use_bf16_inference = use_bf16

    b = chess.Board()
    # warm up
    for _ in range(3):
        mcts_search(net, b, "cpu", sims=sims, history=[], leaf_batch_size=1)

    results = {}
    for K in leaf_batch_sizes:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mcts_search(net, b, "cpu", sims=sims, history=[], leaf_batch_size=K)
            times.append(time.perf_counter() - t0)
        results[K] = {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
        }
    return results


# ─── Parallel throughput sweep ─────────────────────────────────────────────
def _par_worker(q, wid, sims, leaf_batch, use_bf16, n_searches):
    _suppress_threads()
    from mini_az.network import ChessNet
    from mini_az.mcts import mcts_search

    net = ChessNet()
    net.eval()
    net.use_bf16_inference = use_bf16

    b = chess.Board()
    # warm up
    for _ in range(2):
        mcts_search(net, b, "cpu", sims=sims, history=[], leaf_batch_size=leaf_batch)

    t0 = time.perf_counter()
    for _ in range(n_searches):
        mcts_search(net, b, "cpu", sims=sims, history=[], leaf_batch_size=leaf_batch)
    elapsed = time.perf_counter() - t0
    q.put((wid, elapsed, n_searches))


def bench_parallel(worker_counts, sims=200, leaf_batch=8, use_bf16=True,
                   searches_per_worker=3, timeout=180):
    results = {}
    for nw in worker_counts:
        q = mp.Queue()
        procs = []
        for i in range(nw):
            p = mp.Process(target=_par_worker,
                           args=(q, i, sims, leaf_batch, use_bf16, searches_per_worker))
            p.start()
            procs.append(p)

        wall_t0 = time.perf_counter()
        for p in procs:
            p.join(timeout=timeout)
        wall_elapsed = time.perf_counter() - wall_t0

        wr = []
        while not q.empty():
            wr.append(q.get_nowait())
        if not wr:
            results[nw] = None
            continue

        total = sum(r[2] for r in wr)
        wtimes = [r[1] for r in wr]
        tp = total / wall_elapsed
        results[nw] = {
            "workers_ok": len(wr),
            "throughput": tp,
            "avg_ms": np.mean(wtimes) / searches_per_worker * 1000,
            "max_ms": max(wtimes) / searches_per_worker * 1000,
            "wall_s": wall_elapsed,
        }
    return results


# ─── Training throughput sweep ─────────────────────────────────────────────
def bench_training(thread_counts, batch_sizes=(256, 512, 1024), steps=20):
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

    buf = ReplayBuffer(20000)
    samples = [Sample(t.numpy(), fs, ts, pr, pi, 0.0, wdl, 40.0) for _ in range(4000)]
    buf.add_game(samples)

    results = {}
    for nt in thread_counts:
        for bs in batch_sizes:
            torch.set_num_threads(nt)
            net = ChessNet()
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

            # warm up
            for _ in range(3):
                batch = buf.sample_batch(bs)
                coll = collate(batch, "cpu")
                train_step(net, opt, coll)

            t0 = time.perf_counter()
            for _ in range(steps):
                batch = buf.sample_batch(bs)
                coll = collate(batch, "cpu")
                train_step(net, opt, coll)
            elapsed = time.perf_counter() - t0

            results[(nt, bs)] = {
                "steps_per_s": steps / elapsed,
                "ms_per_step": elapsed / steps * 1000,
            }
    return results


# ─── SF teacher latency ───────────────────────────────────────────────────
def bench_sf():
    try:
        from mini_az.stockfish import open_stockfish_engine, sf_teacher_policy_legal
        sf = open_stockfish_engine(
            stockfish_path="/usr/games/stockfish", threads=1, hash_mb=16,
            elo=2000, skill=None,
        )
    except Exception as e:
        return {"error": str(e)}

    b = chess.Board()
    legals = list(b.legal_moves)
    n = 20

    results = {}
    for depth in (8, 12):
        t0 = time.perf_counter()
        for _ in range(n):
            sf_teacher_policy_legal(
                sf, b, legals,
                movetime_ms=0, depth=depth, multipv=5,
                mate_cp=10000, cp_cap=600, cp_soft_scale=150.0, eps=0.02,
            )
        results[f"depth_{depth}_ms"] = (time.perf_counter() - t0) / n * 1000
    sf.quit()
    return results


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    mp.set_start_method("spawn", force=True)
    ncpu = os.cpu_count() or 4
    has_bf16 = _has_bf16_hw()

    print(f"CPU: {ncpu} logical cores")
    print(f"PyTorch: {torch.__version__}")
    print(f"Native BF16 HW: {has_bf16}")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"CPU model: {line.split(':')[1].strip()}")
                    break
    except Exception:
        pass
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 1: bf16 vs fp32 (single-core)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("ETAP 1: BF16 vs FP32 SINGLE-CORE INFERENCE")
    print("=" * 70)

    fp32 = bench_dtype(use_bf16=False, sims=64, repeats=10)
    bf16 = bench_dtype(use_bf16=True, sims=64, repeats=10)
    speedup_bf16 = fp32["mean_ms"] / bf16["mean_ms"]

    print(f"  FP32  64 sims: {fp32['mean_ms']:7.1f} ms  (±{fp32['std_ms']:.1f})")
    print(f"  BF16  64 sims: {bf16['mean_ms']:7.1f} ms  (±{bf16['std_ms']:.1f})")
    print(f"  Speedup BF16:  {speedup_bf16:.2f}x")

    best_bf16 = speedup_bf16 > 1.05  # use bf16 only if actually faster
    print(f"  >>> Use BF16: {best_bf16}")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 2: leaf-batch sweep (single-core, 200 sims)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("ETAP 2: LEAF-BATCH SWEEP (single-core, 200 sims)")
    print("=" * 70)

    lb_sizes = [1, 2, 4, 8, 12, 16]
    lb = bench_leaf_batch(lb_sizes, use_bf16=best_bf16, sims=200, repeats=6)

    baseline_ms = lb[1]["mean_ms"]
    best_K = 1
    best_K_ms = baseline_ms
    print(f"  {'K':>4} {'Mean ms':>10} {'Std ms':>8} {'Speedup':>8}")
    for K in lb_sizes:
        r = lb[K]
        sp = baseline_ms / r["mean_ms"]
        if r["mean_ms"] < best_K_ms:
            best_K_ms = r["mean_ms"]
            best_K = K
        print(f"  {K:>4} {r['mean_ms']:>9.1f} {r['std_ms']:>7.1f} {sp:>7.2f}x")

    print(f"  >>> Best leaf_batch_size: K={best_K} ({baseline_ms/best_K_ms:.2f}x vs K=1)")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 3: parallel worker sweep (with best dtype + leaf batch)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"ETAP 3: PARALLEL WORKERS (sims=200, K={best_K}, bf16={best_bf16})")
    print("=" * 70)

    if ncpu >= 90:
        w_counts = [32, 48, 56, 64, 72, 80, 84, 88, 92]
    elif ncpu >= 16:
        w_counts = [4, 8, 12, ncpu // 2, ncpu - 4, ncpu - 2, ncpu]
    else:
        w_counts = [1, 2, 4, max(4, ncpu - 1), ncpu]

    par = bench_parallel(w_counts, sims=200, leaf_batch=best_K,
                         use_bf16=best_bf16, searches_per_worker=3)

    best_nw = 0
    best_tp = 0
    print(f"  {'Workers':>7} {'Throughput':>12} {'Avg ms':>8} {'Max ms':>8} {'Wall s':>7}")
    for nw in w_counts:
        r = par.get(nw)
        if r is None:
            print(f"  {nw:>7} {'FAILED':>12}")
            continue
        if r["throughput"] > best_tp:
            best_tp = r["throughput"]
            best_nw = nw
        print(f"  {nw:>7} {r['throughput']:>10.1f}/s {r['avg_ms']:>7.0f} "
              f"{r['max_ms']:>7.0f} {r['wall_s']:>6.1f}")
    print(f"  >>> Best: {best_nw} workers = {best_tp:.1f} searches/s")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 4: training sweep (threads x batch size)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("ETAP 4: TRAINING THROUGHPUT (threads x batch_size)")
    print("=" * 70)

    if ncpu >= 90:
        t_counts = [8, 16, 24, 32, 48, 64]
    elif ncpu >= 16:
        t_counts = [4, 8, ncpu // 2, ncpu]
    else:
        t_counts = [2, 4, ncpu]

    batch_sizes = (256, 512, 1024)
    tr = bench_training(t_counts, batch_sizes=batch_sizes, steps=20)

    best_nt = 0
    best_bs = 512
    best_sps = 0
    print(f"  {'Threads':>7} {'Batch':>6} {'Steps/s':>10} {'ms/step':>10}")
    for nt in t_counts:
        for bs in batch_sizes:
            r = tr.get((nt, bs))
            if r is None:
                continue
            if r["steps_per_s"] > best_sps:
                best_sps = r["steps_per_s"]
                best_nt = nt
                best_bs = bs
            print(f"  {nt:>7} {bs:>6} {r['steps_per_s']:>9.1f} {r['ms_per_step']:>9.0f}")
    print(f"  >>> Best: {best_nt} threads, batch={best_bs} = {best_sps:.1f} steps/s")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 5: SF teacher
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("ETAP 5: STOCKFISH TEACHER LATENCY")
    print("=" * 70)
    sf = bench_sf()
    sf_d8_ms = 0
    if "error" in sf:
        print(f"  Stockfish not available: {sf['error']}")
    else:
        for k, v in sf.items():
            print(f"  {k}: {v:.1f} ms/call")
        sf_d8_ms = sf.get("depth_8_ms", 0)
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # COMBINED: estimate iteration time & recommendations
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("COMBINED ESTIMATE & RECOMMENDATIONS")
    print("=" * 70)

    avg_game_plies = 70
    games_per_iter = 60
    steps_per_iter = 200

    # Selfplay wall time (async workers)
    if best_tp > 0:
        sp_wall_s = games_per_iter * avg_game_plies / best_tp
    else:
        sp_wall_s = float("inf")

    # Training wall time (async, overlaps selfplay in Etap 3.1)
    if best_sps > 0:
        tr_wall_s = steps_per_iter / best_sps
    else:
        tr_wall_s = float("inf")

    # With async trainer (Etap 3.1), iter time ≈ max(selfplay, training)
    iter_async_s = max(sp_wall_s, tr_wall_s)
    # Without (sequential): iter time = selfplay + training
    iter_seq_s = sp_wall_s + tr_wall_s

    budget_s = 4 * 3600  # 4 hours
    iters_async = budget_s / iter_async_s if iter_async_s > 0 else 0
    iters_seq = budget_s / iter_seq_s if iter_seq_s > 0 else 0

    print(f"  Selfplay ({games_per_iter} games, {best_nw}w): ~{sp_wall_s:.0f}s")
    print(f"  Training ({steps_per_iter} steps, {best_nt}t): ~{tr_wall_s:.0f}s")
    print()
    print(f"  Sequential iter time:          ~{iter_seq_s:.0f}s")
    print(f"  Async iter time (Etap 3.1+):   ~{iter_async_s:.0f}s")
    print(f"  Iters in 4h (sequential):      ~{iters_seq:.0f}")
    print(f"  Iters in 4h (async):           ~{iters_async:.0f}")
    print()

    # Check if workers need adjustment for async trainer
    # Trainer uses best_nt threads, selfplay workers each use 1 thread
    # Total threads needed: best_nw (workers) + best_nt (trainer) + ~4 (orchestrator/eval)
    total_threads = best_nw + best_nt + 4
    if total_threads > ncpu:
        # Reduce workers to leave room for trainer
        adj_workers = max(ncpu - best_nt - 4, ncpu // 2)
        print(f"  WARNING: {best_nw}w + {best_nt}t + 4 = {total_threads} > {ncpu} cores")
        print(f"  Adjusted workers for async mode: {adj_workers}")
        rec_workers = adj_workers
    else:
        rec_workers = best_nw

    bf16_flag = "--bf16_inference" if best_bf16 else ""

    print()
    print("  ──── RECOMMENDED train_fresh_v3.sh PARAMS ────")
    print(f"  --workers {rec_workers}")
    print(f"  --mp_leaf_batch {best_K}")
    print(f"  --batch {best_bs}")
    print(f"  {bf16_flag}" if bf16_flag else "  (no --bf16_inference)")
    print(f"  --mp_sims 200")
    print(f"  --games_per_iter {games_per_iter}")
    print(f"  --steps_per_iter {steps_per_iter}")
    print()

    # Stage-by-stage speedup summary
    print("  ──── STAGE-BY-STAGE SPEEDUP SUMMARY ────")
    print(f"  Etap 1 (bf16):         {speedup_bf16:.2f}x single-core")
    if best_K > 1:
        print(f"  Etap 2 (leaf K={best_K}):    {baseline_ms/best_K_ms:.2f}x single-core (200 sims)")
    print(f"  Etap 3 (async):        ~{iter_seq_s/iter_async_s:.2f}x iteration throughput")
    combined_sc = fp32["mean_ms"] / best_K_ms if best_K > 1 else speedup_bf16
    print(f"  Combined (E1+E2):      ~{combined_sc:.2f}x single-core")
    print()


if __name__ == "__main__":
    main()
