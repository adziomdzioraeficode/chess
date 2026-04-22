#!/usr/bin/env python3
"""Worker-count sweep: find optimal balance for D96s v6 (48 phys / 96 HT).

Tests training speed (accum=2, batch=1024, 24 trainer threads)
with different numbers of concurrent selfplay workers.
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
import torch
import chess


def busy_worker(stop_ev, wid):
    """Simulate a selfplay worker: continuous MCTS on 1 thread."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    from mini_az.network import ChessNet
    from mini_az.mcts import mcts_search
    net = ChessNet()
    net.eval()
    b = chess.Board()
    while not stop_ev.is_set():
        mcts_search(net, b, "cpu", sims=200, history=[])


def make_buffer(n=5000):
    from mini_az.encoding import board_to_tensor, legal_moves_canonical
    from mini_az.training import Sample, ReplayBuffer

    b = chess.Board()
    t = board_to_tensor(b, history=[])
    legals = legal_moves_canonical(b)
    fs = np.array([m[0] for m in legals], dtype=np.int64)
    ts = np.array([m[1] for m in legals], dtype=np.int64)
    pr = np.array([m[2] for m in legals], dtype=np.int64)
    pi = np.ones(len(legals), dtype=np.float32) / len(legals)
    wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    buf = ReplayBuffer(100000)
    for _ in range(n):
        z = float(np.random.choice([-1, 0, 0, 0, 1]))
        buf.add_game([Sample(t.numpy(), fs, ts, pr, pi, z, wdl, 40.0)])
    return buf


def bench_with_workers(buf, n_workers, batch_size=1024, accum=2, steps=12, n_threads=24):
    from mini_az.network import ChessNet
    from mini_az.training import collate, train_step

    torch.set_num_threads(n_threads)
    net = ChessNet()
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    # Warmup
    for _ in range(3):
        batch = collate(buf.sample_batch(batch_size), "cpu")
        train_step(net, opt, batch)

    # Launch workers
    stop = mp.Event()
    procs = []
    if n_workers > 0:
        for i in range(n_workers):
            p = mp.Process(target=busy_worker, args=(stop, i))
            p.start()
            procs.append(p)
        time.sleep(6)  # let them stabilize

    loss_scale = 1.0 / accum
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        for _ in range(accum):
            batch_s = buf.sample_batch_mixed(
                batch_size, recent_frac=0.7, recent_window=3000, decisive_frac=0.15
            )
            batch = collate(batch_s, "cpu")
            train_step(net, opt, batch, loss_scale=loss_scale, do_step=False)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
    elapsed = time.perf_counter() - t0
    ms = elapsed / steps * 1000

    # Cleanup
    if procs:
        stop.set()
        for p in procs:
            p.join(timeout=15)
        alive = sum(1 for p in procs if p.is_alive())
        if alive:
            for p in procs:
                if p.is_alive():
                    p.terminate()

    return ms


def main():
    mp.set_start_method("spawn", force=True)
    ncpu = os.cpu_count() or 4
    print(f"CPU: {ncpu} cores | PyTorch: {torch.__version__}")
    print(f"Config: batch=1024, accum=2, trainer_threads=24")
    print()

    buf = make_buffer(5000)

    worker_counts = [0, 32, 48, 56, 64, 68, 72]
    print(f"{'Workers':>8} {'ms/step':>10} {'Proj 40 steps':>15} {'vs alone':>10}")
    print("-" * 50)

    baseline_ms = None
    for nw in worker_counts:
        ms = bench_with_workers(buf, nw, steps=12)
        proj = ms * 40 / 1000
        if baseline_ms is None:
            baseline_ms = ms
        ratio = ms / baseline_ms
        marker = " ← current" if nw == 68 else ""
        print(f"{nw:>8} {ms:>9.0f} {proj:>12.1f}s {ratio:>9.2f}x{marker}")

    print()
    print("Lower is better. 'vs alone' = slowdown factor from CPU contention.")
    print("Current config uses 68 workers. Consider reducing if contention is high.")


if __name__ == "__main__":
    main()
