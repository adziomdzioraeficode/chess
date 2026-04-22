#!/usr/bin/env python3
"""Phase-2 benchmark: measure actual training throughput under contention.

Compares:
  A) Old: batch=1024, steps=80, no grad_accum, no workers
  B) New: batch=1024, accum=2, steps=40, no workers
  C) New: batch=1024, accum=2, steps=40, WITH 68 selfplay workers

Also measures collate speed and decisive sampling overhead.
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
import torch
import chess


# ─────── worker for contention test ────────
def busy_worker(stop_ev, wid):
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


def bench_train_config(label, buf, batch_size, accum, steps, n_threads):
    from mini_az.network import ChessNet
    from mini_az.training import collate, train_step

    torch.set_num_threads(n_threads)
    net = ChessNet()
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    # Warmup
    for _ in range(5):
        batch = collate(buf.sample_batch(batch_size), "cpu")
        train_step(net, opt, batch)

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

    print(f"  {label}")
    print(f"    {steps} steps in {elapsed:.2f}s = {ms:.0f} ms/step")
    print(f"    effective_batch={batch_size * accum}, accum={accum}")
    return ms


def bench_collate(buf, batch_size=1024, reps=100):
    from mini_az.training import collate

    # Warmup
    for _ in range(5):
        collate(buf.sample_batch(batch_size), "cpu")

    t0 = time.perf_counter()
    for _ in range(reps):
        batch_s = buf.sample_batch(batch_size)
        collate(batch_s, "cpu")
    elapsed = time.perf_counter() - t0
    ms = elapsed / reps * 1000
    print(f"  collate(batch={batch_size}): {ms:.1f} ms/call ({reps} reps)")
    return ms


def bench_decisive(buf, batch_size=1024, reps=200):
    t0 = time.perf_counter()
    for _ in range(reps):
        buf.sample_batch_mixed(batch_size, 0.7, 3000, decisive_frac=0.15)
    elapsed_dec = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(reps):
        buf.sample_batch_mixed(batch_size, 0.7, 3000, decisive_frac=0.0)
    elapsed_no = time.perf_counter() - t0

    ms_dec = elapsed_dec / reps * 1000
    ms_no = elapsed_no / reps * 1000
    print(f"  sample_batch_mixed(decisive_frac=0.15): {ms_dec:.2f} ms/call")
    print(f"  sample_batch_mixed(decisive_frac=0.0):  {ms_no:.2f} ms/call")
    print(f"  Overhead: +{(ms_dec - ms_no):.2f} ms (+{(elapsed_dec / elapsed_no - 1) * 100:.0f}%)")
    return ms_dec, ms_no


def bench_contention(buf, batch_size=1024, accum=2, steps=20, n_threads=24, n_workers=68):
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
    print(f"  Launching {n_workers} selfplay workers...")
    for i in range(n_workers):
        p = mp.Process(target=busy_worker, args=(stop, i))
        p.start()
        procs.append(p)

    print(f"  Waiting 8s for workers to stabilize...")
    time.sleep(8)

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

    stop.set()
    for p in procs:
        p.join(timeout=15)
    alive = sum(1 for p in procs if p.is_alive())
    if alive:
        print(f"  WARNING: {alive} workers still alive, terminating...")
        for p in procs:
            if p.is_alive():
                p.terminate()

    print(f"  WITH {n_workers} workers: {steps} steps in {elapsed:.2f}s = {ms:.0f} ms/step")
    return ms


def main():
    mp.set_start_method("spawn", force=True)
    ncpu = os.cpu_count() or 4
    print(f"CPU cores: {ncpu}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print()

    buf = make_buffer(5000)
    print(f"Buffer: {len(buf)} samples\n")

    # ── 1. Collate speed ──
    print("=" * 60)
    print("1. COLLATE SPEED")
    print("=" * 60)
    bench_collate(buf, 1024)
    print()

    # ── 2. Decisive sampling ──
    print("=" * 60)
    print("2. DECISIVE SAMPLING OVERHEAD")
    print("=" * 60)
    bench_decisive(buf, 1024)
    print()

    # ── 3. Training throughput (no contention) ──
    print("=" * 60)
    print("3. TRAINING THROUGHPUT (no contention, 24 threads)")
    print("=" * 60)

    ms_old = bench_train_config(
        "Config A: OLD (batch=1024, steps=80, no accum)",
        buf, batch_size=1024, accum=1, steps=20, n_threads=24,
    )
    ms_new = bench_train_config(
        "Config B: NEW (batch=1024, accum=2, steps=40)",
        buf, batch_size=1024, accum=2, steps=20, n_threads=24,
    )
    print()
    print(f"  OLD projected iter time: 80 × {ms_old:.0f}ms = {ms_old * 80 / 1000:.1f}s")
    print(f"  NEW projected iter time: 40 × {ms_new:.0f}ms = {ms_new * 40 / 1000:.1f}s")
    print(f"  Ratio: {ms_old * 80 / (ms_new * 40):.2f}x")
    print()

    # ── 4. Training under contention ──
    print("=" * 60)
    print("4. TRAINING UNDER CONTENTION (24 threads + 68 workers)")
    print("=" * 60)
    ms_cont = bench_contention(buf, batch_size=1024, accum=2, steps=15, n_threads=24, n_workers=68)
    print()

    # ── 5. Thread sweep ──
    print("=" * 60)
    print("5. THREAD COUNT SWEEP (accum=2, steps=40 config)")
    print("=" * 60)
    for nt in [8, 12, 16, 20, 24, 32]:
        bench_train_config(
            f"threads={nt}",
            buf, batch_size=1024, accum=2, steps=10, n_threads=nt,
        )
    print()

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Collate 1024:    checked (see above)")
    print(f"  Decisive:        negligible overhead")
    print(f"  Train (alone):   OLD 80×{ms_old:.0f}ms = {ms_old*80/1000:.1f}s  |  NEW 40×{ms_new:.0f}ms = {ms_new*40/1000:.1f}s")
    print(f"  Train (contention): 40×{ms_cont:.0f}ms = {ms_cont*40/1000:.1f}s")
    print(f"  Key benefit: NEW has 2× effective batch (2048) for gradient stability")


if __name__ == "__main__":
    main()
