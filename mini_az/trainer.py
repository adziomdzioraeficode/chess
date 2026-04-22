"""Async trainer process.

Runs in its own process alongside the self-play workers. Owns the network,
optimizer, scheduler, and replay buffer. Each "iteration" drains
`args.games_per_iter` games from the shared `out_q` into its buffer, runs
`args.steps_per_iter` gradient steps, publishes updated weights via
`broadcast_weights` (shm path + version counter), optionally saves a
snapshot, and reports per-iter metrics back to the orchestrator through
`metrics_q`.

The trainer respects `pause_ev`: during eval the orchestrator sets it so both
workers and the trainer release CPU. Between iterations the workers never
pause for training — that's the main throughput win over the old synchronous
loop where `pause_ev.set()` idled 92 worker cores for the full training phase.
"""

import gzip
import os
import pickle
import queue
import shutil
import time
import warnings

import torch

from .config import print
from .checkpoint import load_checkpoint
from .network import ChessNet
from .selfplay import broadcast_weights
from .training import ReplayBuffer, collate, train_step
from .train_helpers import (
    init_counters, accumulate_info, print_selfplay_info, print_z_stats, save_iter,
)


def trainer_loop(
    args,
    out_q,
    metrics_q,
    weights_path: str,
    weights_version,
    best_path: str,
    stop_ev,
    pause_ev,
    device: str,
):
    torch.set_num_threads(int(getattr(args, "trainer_threads", 0)) or min(24, max(8, args.workers)))
    torch.set_num_interop_threads(2)

    # --- Model + checkpoint ---
    net = ChessNet().to(device)
    # Trainer itself doesn't run inference at batch=1, but workers reload
    # its weights and do — set the flag here so initial-iteration workers
    # have a consistent env (no effect on training math, which stays fp32).
    net.use_bf16_inference = bool(getattr(args, "bf16_inference", False))
    if os.path.exists(args.model):
        try:
            net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
            print(f"[trainer] loaded model weights: {args.model}")
        except Exception as e:
            print(f"[trainer] could not load {args.model}: {e}. Starting fresh.")
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Replay buffer ---
    if args.replay_path and os.path.exists(args.replay_path):
        try:
            rb = ReplayBuffer.load(args.replay_path).resize(args.buffer)
            print(f"[trainer] loaded replay buffer: {args.replay_path} (len={len(rb)} maxlen={rb.maxlen})")
        except (gzip.BadGzipFile, EOFError, pickle.UnpicklingError, OSError) as e:
            ts = time.strftime("%Y%m%d-%H%M%S")
            corrupt = f"{args.replay_path}.corrupt.{ts}"
            try:
                shutil.move(args.replay_path, corrupt)
                print(f"[trainer] WARNING: replay buffer corrupted ({e}). Moved to: {corrupt}")
            except Exception as e2:
                print(f"[trainer] WARNING: replay buffer corrupted ({e}). Also failed to move: {e2}")
            rb = ReplayBuffer(maxlen=int(args.buffer))
            print(f"[trainer] created new replay buffer (len={len(rb)} maxlen={rb.maxlen})")
    else:
        rb = ReplayBuffer(maxlen=int(args.buffer))
        print(f"[trainer] created new replay buffer (len={len(rb)} maxlen={rb.maxlen})")

    if args.clear_buffer:
        old_len = len(rb)
        rb = ReplayBuffer(maxlen=int(args.buffer))
        print(f"[trainer] --clear_buffer: discarded {old_len} samples, fresh buffer")

    start_iter = 0
    if args.ckpt and os.path.exists(args.ckpt):
        try:
            start_iter = load_checkpoint(args.ckpt, net, opt, device=device, load_opt=args.resume_opt)
            print(f"[trainer] loaded checkpoint: {args.ckpt} (iter={start_iter})")
        except Exception as e:
            print(f"[trainer] could not load checkpoint ({e}).")

    # Reset LR (see main.py rationale).
    for pg in opt.param_groups:
        pg['lr'] = args.lr
        pg.pop('initial_lr', None)
    remaining_iters = max(1, args.iters - start_iter)
    total_steps = remaining_iters * args.steps_per_iter
    warmup_steps = min(500, max(1, total_steps // 10))
    cosine_steps = total_steps - warmup_steps
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cosine_steps), eta_min=args.lr * 0.01
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*epoch parameter.*deprecated.*")
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
        )
    print(f"[trainer] LR schedule: warmup {warmup_steps} → cosine {cosine_steps} "
          f"(base={args.lr} eta_min={args.lr*0.01:.2e})")

    # Signal start to orchestrator so it knows the initial iter number.
    try:
        metrics_q.put({"kind": "ready", "start_iter": start_iter}, timeout=5)
    except queue.Full:
        pass

    def _wait_if_paused():
        """Block while pause_ev is set (eval is running). Cooperative."""
        while pause_ev.is_set() and not stop_ev.is_set():
            time.sleep(0.25)

    # --- Main iteration loop ---
    try:
        for it in range(start_iter + 1, args.iters + 1):
            if stop_ev.is_set():
                break
            _wait_if_paused()
            if stop_ev.is_set():
                break

            t_iter0 = time.time()
            t_sp0 = time.time()
            c = init_counters()
            res_counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0, "other": 0}
            games_collected = 0
            new_samples = 0
            plies_sum = 0

            # Drain games from workers. Workers keep playing the whole time —
            # this loop no longer pauses them.
            while games_collected < args.games_per_iter:
                if stop_ev.is_set():
                    return
                _wait_if_paused()
                try:
                    samples, res, plies, info = out_q.get(timeout=10)
                except queue.Empty:
                    print(f"[trainer] iter {it}: waiting for games...")
                    continue
                rb.add_game(samples)
                accumulate_info(info, c, args)
                games_collected += 1
                new_samples += len(samples)
                plies_sum += int(plies)
                res_counts[res] = res_counts.get(res, 0) + 1

            t_sp = time.time() - t_sp0
            avg_plies = plies_sum / max(1, games_collected)

            # --- Training ---
            t_tr0 = time.time()
            net.train()
            last_m = None
            accum = max(1, int(getattr(args, "grad_accum", 1)))
            loss_scale = 1.0 / accum
            for step in range(args.steps_per_iter):
                _wait_if_paused()
                if stop_ev.is_set():
                    return
                if len(rb) < args.batch:
                    continue
                # Gradient accumulation: accumulate over `accum` mini-batches
                # then do one optimizer step (effective batch = batch * accum).
                opt.zero_grad()
                for _sub in range(accum):
                    batch_s = rb.sample_batch_mixed(
                        args.batch, recent_frac=args.recent_frac,
                        recent_window=args.recent_window,
                        sharp_frac=args.sharp_frac,
                        sharp_threshold=args.sharp_threshold,
                        decisive_frac=getattr(args, "decisive_frac", 0.0),
                    )
                    batch = collate(batch_s, device)
                    m = train_step(net, opt, batch, val_w=args.val_w,
                                   moves_left_w=args.moves_left_w,
                                   loss_scale=loss_scale, do_step=False)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0))
                opt.step()
                if m is not None:
                    m["grad_norm"] = grad_norm
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*epoch parameter.*deprecated.*")
                    scheduler.step()
                last_m = m
            t_tr = time.time() - t_tr0
            cur_lr = scheduler.get_last_lr()[0]

            # --- Save snapshot / opponent update ---
            if args.save_every and (it % args.save_every == 0):
                save_iter(args, net, opt, it, rb, best_path)

            # --- Broadcast new weights to workers ---
            broadcast_weights(weights_path, net, weights_version)

            # --- Diagnostics (printed inside trainer for immediacy) ---
            if games_collected:
                print_selfplay_info(it, c, games_collected)
            if len(rb) >= 1000:
                print_z_stats(it, rb)

            # --- Report to orchestrator ---
            t_iter = time.time() - t_iter0
            try:
                metrics_q.put({
                    "kind": "iter",
                    "iter": it,
                    "games_collected": games_collected,
                    "new_samples": new_samples,
                    "rb_len": len(rb),
                    "res_counts": res_counts,
                    "avg_plies": avg_plies,
                    "counters": c,
                    "last_m": last_m,
                    "t_sp": t_sp,
                    "t_tr": t_tr,
                    "t_iter": t_iter,
                    "cur_lr": cur_lr,
                }, timeout=5)
            except queue.Full:
                print(f"[trainer] WARN: metrics_q full at iter {it}, dropping")

            print(f"[trainer] iter {it}: selfplay={t_sp:.1f}s train={t_tr:.1f}s total={t_iter:.1f}s "
                  f"games={games_collected} rb={len(rb)}")

    finally:
        # Final weight publish so eval can see the last updates.
        try:
            broadcast_weights(weights_path, net, weights_version)
        except Exception:
            pass
        try:
            metrics_q.put({"kind": "done"}, timeout=2)
        except Exception:
            pass
        print("[trainer] exiting")
