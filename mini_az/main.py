"""Main entry point: argument parsing, training loop, eval, UCI."""

import argparse
import gzip
import os
import pickle
import queue
import shutil
import sys
import time
import warnings
import multiprocessing as mp
from multiprocessing.process import BaseProcess
from datetime import datetime, timezone
from typing import List

import numpy as np
import chess
import torch
from tqdm import trange

from .config import print
from .encoding import INPUT_PLANES, board_to_tensor, append_eval_csv
from .network import ChessNet
from .mcts import mcts_search
from .training import ReplayBuffer, collate, train_step
from .checkpoint import save_checkpoint, load_checkpoint
from .stockfish import open_stockfish_engine
from .selfplay import (
    make_game_samples_unified, selfplay_worker,
    broadcast_weights, broadcast_weights_initial, resolve_device,
    default_weights_shm_path,
)
from .eval_play import (
    play_vs_stockfish, play_vs_random, play_vs_model, uci_loop,
)
from .distill import run_distillation
from .trainer import trainer_loop
from .evaluator import run_eval_job
from .train_helpers import (
    init_counters, accumulate_info, print_selfplay_info, print_z_stats,
    log_iteration_csv, save_iter,
)


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "uci", "distill"], required=True)
    # Distillation-specific args
    ap.add_argument("--distill_samples", type=int, default=500_000,
                    help="Target number of SF-labeled positions for distillation")
    ap.add_argument("--distill_epochs", type=int, default=3)
    ap.add_argument("--distill_eval_depth", type=int, default=12,
                    help="SF depth for value evaluation in distillation")
    ap.add_argument("--distill_policy_depth", type=int, default=10,
                    help="SF depth for policy (multipv) in distillation")
    ap.add_argument("--distill_multipv", type=int, default=8)
    ap.add_argument("--distill_positions_per_game", type=int, default=10)
    ap.add_argument("--distill_max_random_plies", type=int, default=120)
    ap.add_argument("--resume_opt", action="store_true", help="Also resume optimizer state")
    ap.add_argument("--max_plies", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--eval_csv", default="eval_log.csv")
    ap.add_argument("--sf_elo", type=int, default=2000)
    ap.add_argument("--sf_eval_elo", type=int, default=1320)
    ap.add_argument("--sf_eval_elo_easy", type=int, default=-1,
                    help="If >=0, also eval vs SF Skill Level (0-20, not Elo)")

    ap.add_argument("--mcts_value_mix", type=float, default=0.0)
    ap.add_argument("--mix_best", type=float, default=0.30)
    ap.add_argument("--opponent_model", default="models/opponent.pt")
    ap.add_argument("--mix_opp", type=float, default=0.20)
    ap.add_argument("--opp_reload_sec", type=float, default=60)
    ap.add_argument("--no_sf", action="store_true", help="Disable Stockfish bootstrap in workers")
    ap.add_argument("--sf_worker_frac", type=float, default=0.50,
                    help="Fraction of self-play workers with local Stockfish enabled")

    ap.add_argument("--opp_lag", type=int, default=10)

    ap.add_argument("--best_reload_sec", type=float, default=60)
    ap.add_argument("--best_model", default="models/best.pt")
    ap.add_argument("--gate_margin", type=float, default=0.005)

    ap.add_argument("--val_w", type=float, default=1.0)
    ap.add_argument("--resign_threshold", type=float, default=-0.95)
    ap.add_argument("--resign_patience", type=int, default=10)

    ap.add_argument("--rand_eval_games", type=int, default=6)
    ap.add_argument("--rand_eval_sims", type=int, default=32)
    ap.add_argument("--rand_max_plies", type=int, default=120)
    ap.add_argument("--sf_eval_max_plies", type=int, default=220)

    ap.add_argument("--save_dir", default="models")
    ap.add_argument("--save_every", type=int, default=10)

    ap.add_argument("--model", default="mini_az.pt")
    ap.add_argument("--ckpt", default="mini_az_ckpt.pt")
    ap.add_argument("--replay_path", default="replay.pkl.gz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games_per_iter", type=int, default=200)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps_per_iter", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=500_000)

    ap.add_argument("--workers", type=int, default=80)
    ap.add_argument("--mp_sims", type=int, default=64)
    ap.add_argument("--mp_leaf_batch", type=int, default=1,
                    help="Virtual-loss leaf batch size for worker MCTS (>=1, default 1 = sequential).")
    ap.add_argument("--trainer_threads", type=int, default=0,
                    help="Thread count for the async trainer process (0 = auto, min(24, max(8, workers))).")
    ap.add_argument("--eval_threads", type=int, default=0,
                    help="Thread count for the async eval process (0 = auto, max(4, workers//12)).")
    ap.add_argument("--bf16_inference", action="store_true",
                    help="Use bf16 autocast in MCTS inference. ONLY enable on AVX-512-BF16/AMX "
                         "hardware (EPYC 9004+, Xeon SPR+) — on older CPUs PyTorch emulates "
                         "bf16 and forward is ~13x slower.")

    ap.add_argument("--sf_path", default="stockfish")
    ap.add_argument("--sf_skill", type=int, default=0)
    ap.add_argument("--sf_movetime_ms", type=int, default=30)
    ap.add_argument("--eval_games", type=int, default=10)
    ap.add_argument("--eval_sims", type=int, default=100)

    ap.add_argument("--self_eval_games", type=int, default=10)
    ap.add_argument("--self_eval_sims", type=int, default=32)
    ap.add_argument("--self_eval_max_plies", type=int, default=120)

    ap.add_argument("--recent_frac", type=float, default=0.6)
    ap.add_argument("--recent_window", type=int, default=300_000)
    ap.add_argument("--sharp_frac", type=float, default=0.20)
    ap.add_argument("--sharp_threshold", type=float, default=0.35)

    ap.add_argument("--sf_boot_prob", type=float, default=1.0)
    ap.add_argument("--sf_boot_time_ms", type=int, default=20)
    ap.add_argument("--sf_cp_scale", type=float, default=800.0)
    ap.add_argument("--sf_cp_cap", type=int, default=1200)
    ap.add_argument("--sf_boot_depth", type=int, default=6)
    ap.add_argument("--sf_teacher_prob", type=float, default=0.5)
    ap.add_argument("--sf_teacher_mix", type=float, default=0.35)
    ap.add_argument("--sf_teacher_time_ms", type=int, default=0)
    ap.add_argument("--sf_teacher_depth", type=int, default=8)
    ap.add_argument("--sf_teacher_multipv", type=int, default=4)
    ap.add_argument("--sf_teacher_cp_cap", type=int, default=800)
    ap.add_argument("--sf_teacher_cp_soft_scale", type=float, default=210.0)
    ap.add_argument("--sf_teacher_eps", type=float, default=0.01)

    ap.add_argument("--clear_buffer", action="store_true",
                    help="Discard replay buffer on startup (keep model weights)")
    ap.add_argument("--moves_left_w", type=float, default=0.15)

    args = ap.parse_args()
    # Dump all args for reproducibility
    print("=" * 60)
    print(f"mini_az_chess  started at {datetime.now(timezone.utc).isoformat()}")
    print(f"mode={args.mode}  device_req={args.device}  pid={os.getpid()}")
    for k, v in sorted(vars(args).items()):
        print(f"  --{k}={v}")
    print("=" * 60)

    device = resolve_device(args.device)

    net = ChessNet().to(device)
    net.use_bf16_inference = bool(args.bf16_inference)
    if args.bf16_inference:
        print("[setup] bf16 inference autocast: ENABLED (expect ~2x on AVX-512-BF16/AMX hosts).")

    if os.path.exists(args.model):
        try:
            net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
            print(f"Loaded model weights: {args.model}")
        except Exception as e:
            print(f"Could not load model weights ({e}). Starting fresh.")

    def _ensure_snapshot(path: str, label: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        recreate = False
        if os.path.exists(path):
            try:
                probe = ChessNet()
                probe.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            except Exception as e:
                print(f"{label} incompatible ({e}). Reinitializing: {path}")
                recreate = True
        else:
            recreate = True

        if recreate:
            tmp = path + f".tmp.{os.getpid()}"
            torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, tmp)
            os.replace(tmp, path)
            print(f"Initialized {label.lower()}: {path}")

    best_path = args.best_model
    _ensure_snapshot(best_path, "Best model")

    opponent_path = args.opponent_model
    _ensure_snapshot(opponent_path, "Opponent model")

    if args.mode == "train":
        _run_train(args, net, device, best_path)
    elif args.mode == "distill":
        run_distillation(
            net, device,
            sf_path=args.sf_path,
            sf_elo=args.sf_elo,
            workers=args.workers if args.workers > 0 else 80,
            target_samples=args.distill_samples,
            batch_size=args.batch,
            lr=args.lr,
            epochs=args.distill_epochs,
            sf_eval_depth=args.distill_eval_depth,
            sf_policy_depth=args.distill_policy_depth,
            sf_multipv=args.distill_multipv,
            sf_cp_scale=args.sf_cp_scale,
            sf_cp_cap=args.sf_cp_cap,
            positions_per_game=args.distill_positions_per_game,
            max_random_plies=args.distill_max_random_plies,
            val_w=args.val_w,
            moves_left_w=args.moves_left_w,
            save_path=args.model,
        )
    elif args.mode == "eval":
        net.eval()
        score, winrate, elo_diff = play_vs_stockfish(
            net, device,
            stockfish_path=args.sf_path, sf_skill=args.sf_skill,
            sf_movetime_ms=args.sf_movetime_ms, games=args.eval_games,
            sims=args.eval_sims, sf_elo=args.sf_elo, max_plies=args.sf_eval_max_plies
        )
        rnd_score, rnd_winrate, rnd_elo_diff = play_vs_random(
            net, device, games=args.rand_eval_games,
            sims=args.rand_eval_sims, max_plies=args.rand_max_plies
        )
        print(f"Score vs Random: {rnd_score:.3f}")
        print(f"Winrate vs Random: {rnd_winrate:.3f}")
        print(f"Estimated Elo diff vs Random: {rnd_elo_diff:+.0f}")
        print(f"Score vs Stockfish(elo={args.sf_elo}, {args.sf_movetime_ms}ms): {score:.3f}")
        print(f"Winrate: {winrate:.3f}")
        print(f"Estimated Elo diff (rough): {elo_diff:+.0f}")
    elif args.mode == "uci":
        net.eval()
        uci_loop(net, device, sims=args.sims)


def _run_train(args, net, device, best_path):
    """Dispatch to orchestrator (async trainer process) or inline single-process
    path depending on args.workers."""
    b = chess.Board()
    t = board_to_tensor(b, history=None)
    assert t.shape == (INPUT_PLANES, 8, 8), f"Expected ({INPUT_PLANES},8,8), got {t.shape}"

    if args.workers > 0:
        _run_train_orchestrator(args, net, device, best_path)
    else:
        _run_train_inline(args, net, device, best_path)


def _run_train_orchestrator(args, net, device, best_path):
    """Async architecture: self-play workers generate games continuously into
    out_q; a separate trainer process consumes them, runs gradient steps and
    publishes fresh weights to /dev/shm. The orchestrator (this function)
    only: spawns children, logs per-iter metrics received from the trainer,
    runs the eval gate (which is the only thing that pauses workers + trainer),
    and handles clean shutdown.
    """
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=max(512, args.games_per_iter * 10))
    metrics_q = ctx.Queue(maxsize=100)
    stop_ev = ctx.Event()
    pause_ev = ctx.Event()
    procs: List[BaseProcess] = []

    weights_path = default_weights_shm_path()
    weights_version = ctx.Value('q', 0)

    # Publish initial weights before spawning anything — workers and trainer
    # both block until version > 0.
    broadcast_weights_initial(weights_path, net, weights_version)
    print(f"Initial weights published to {weights_path} (version={weights_version.value})")

    sf_worker_frac = float(np.clip(args.sf_worker_frac, 0.0, 1.0))
    sf_enabled_workers = min(args.workers, max(0, int(round(args.workers * sf_worker_frac))))
    for wid in range(args.workers):
        enable_sf = (not args.no_sf) and (wid < sf_enabled_workers)
        sf_teacher_depth = None if int(args.sf_teacher_depth) <= 0 else int(args.sf_teacher_depth)
        p = ctx.Process(target=selfplay_worker,
                        args=(wid, weights_path, weights_version, out_q, stop_ev, pause_ev,
                              args.mp_sims, args.max_plies,
                              args.resign_threshold, args.resign_patience,
                              args.best_model, args.mix_best, args.best_reload_sec,
                              args.opponent_model, args.mix_opp, args.opp_reload_sec,
                              args.sf_boot_time_ms, args.sf_boot_prob,
                              args.sf_cp_scale, args.sf_cp_cap, args.sf_boot_depth,
                              args.sf_teacher_prob, args.sf_teacher_mix,
                              args.sf_teacher_time_ms, sf_teacher_depth,
                              args.sf_teacher_multipv, args.sf_teacher_cp_cap,
                              args.sf_teacher_cp_soft_scale, args.sf_teacher_eps,
                              enable_sf, args.sf_elo, args.mcts_value_mix,
                              max(1, int(args.mp_leaf_batch)),
                              bool(args.bf16_inference),
                              ),
                        daemon=True)
        p.start()
        procs.append(p)

    time.sleep(1)
    print("Workers alive:", [p.is_alive() for p in procs])
    print(f"Stockfish-enabled workers: {sf_enabled_workers}/{args.workers} (frac={sf_worker_frac:.2f})")
    print(f"Started {args.workers} self-play workers.")

    # Spawn trainer. It does its own model/opt/scheduler/rb setup, then drains
    # out_q continuously.
    trainer_proc = ctx.Process(
        target=trainer_loop,
        args=(args, out_q, metrics_q, weights_path, weights_version,
              best_path, stop_ev, pause_ev, device),
        daemon=True,
    )
    trainer_proc.start()
    print("Started async trainer process.")

    import queue as _queue

    # Fire-and-forget eval processes. We reap finished ones lazily so the
    # orchestrator never waits for an eval. Workers and trainer keep running
    # the whole time — eval just competes for CPU (sized by --eval_threads).
    active_evals: List[BaseProcess] = []

    def _reap_finished_evals():
        nonlocal active_evals
        still = []
        for ep in active_evals:
            if ep.is_alive():
                still.append(ep)
            else:
                ep.join(timeout=0.1)
        active_evals = still

    try:
        while True:
            if not trainer_proc.is_alive():
                print("[orchestrator] trainer exited — stopping.")
                break
            try:
                msg = metrics_q.get(timeout=30)
            except _queue.Empty:
                alive = [p.is_alive() for p in procs]
                if not any(alive):
                    print("[orchestrator] all workers died.")
                    break
                _reap_finished_evals()
                print(f"[orchestrator] waiting for trainer metrics... "
                      f"workers_alive={sum(alive)}/{len(procs)} trainer_alive={trainer_proc.is_alive()} "
                      f"active_evals={len(active_evals)}")
                continue

            kind = msg.get("kind")
            if kind == "ready":
                si = int(msg.get("start_iter", 0))
                print(f"[orchestrator] trainer ready at iter {si}")
                continue
            if kind == "done":
                print("[orchestrator] trainer signaled done.")
                break
            if kind != "iter":
                continue

            it = int(msg["iter"])
            log_iteration_csv(
                args, it, msg["counters"], msg["games_collected"],
                msg["res_counts"], msg["avg_plies"], msg["new_samples"],
                msg["rb_len"], msg["last_m"], msg["t_sp"], msg["t_tr"],
                msg["t_iter"], msg["cur_lr"],
            )

            _reap_finished_evals()
            if args.eval_every and (it % args.eval_every == 0):
                # Only one eval in flight at a time — skip if previous hasn't
                # finished yet (keeps RAM bounded and avoids duplicate work).
                if active_evals:
                    print(f"[orchestrator] iter {it}: previous eval still running, skipping this eval")
                else:
                    ep = ctx.Process(
                        target=run_eval_job,
                        args=(args, device, it, best_path, weights_path,
                              msg["counters"], msg["games_collected"], msg["rb_len"]),
                        daemon=False,  # allow eval to finish cleanly on shutdown
                    )
                    ep.start()
                    active_evals.append(ep)
                    print(f"[orchestrator] iter {it}: spawned eval process pid={ep.pid}")

            if it >= args.iters:
                print(f"[orchestrator] reached --iters={args.iters}, stopping.")
                break
    finally:
        stop_ev.set()
        pause_ev.clear()

        try:
            out_q.cancel_join_thread()
        except Exception:
            pass
        try:
            metrics_q.cancel_join_thread()
        except Exception:
            pass
        # Drain so workers blocked on put() can exit.
        for _ in range(10000):
            try:
                out_q.get_nowait()
            except _queue.Empty:
                break
            except Exception:
                break

        if trainer_proc.is_alive():
            trainer_proc.join(timeout=30)
            if trainer_proc.is_alive():
                trainer_proc.terminate()
                trainer_proc.join(timeout=5)

        time.sleep(1)
        for p in procs:
            p.join(timeout=0.1)
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(timeout=1)
        print("All workers stopped.")

        # Give in-flight eval processes a chance to finish writing their CSV
        # row and atomic best.pt promotion. If they stall, kill them so the
        # script exits.
        for ep in active_evals:
            ep.join(timeout=60)
            if ep.is_alive():
                print(f"[orchestrator] eval pid={ep.pid} exceeded timeout, terminating")
                ep.terminate()
                ep.join(timeout=5)

        try:
            if os.path.exists(weights_path):
                os.unlink(weights_path)
        except Exception:
            pass


def _run_train_inline(args, net, device, best_path):
    """Single-process synchronous path (--workers 0). Kept for debugging and
    small local runs. Still exercises the full RL loop (MCTS, replay buffer,
    WDL/value heads, Stockfish teacher/bootstrap) but without multiprocessing.
    """
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.replay_path and os.path.exists(args.replay_path):
        try:
            rb = ReplayBuffer.load(args.replay_path).resize(args.buffer)
            print(f"Loaded replay buffer: {args.replay_path} (len={len(rb)} maxlen={rb.maxlen})")
        except (gzip.BadGzipFile, EOFError, pickle.UnpicklingError, OSError) as e:
            ts = time.strftime("%Y%m%d-%H%M%S")
            corrupt = f"{args.replay_path}.corrupt.{ts}"
            try:
                shutil.move(args.replay_path, corrupt)
                print(f"WARNING: replay buffer is corrupted ({e}). Moved to: {corrupt}")
            except Exception as e2:
                print(f"WARNING: replay buffer is corrupted ({e}). Also failed to move file: {e2}")
            rb = ReplayBuffer(maxlen=int(args.buffer))
    else:
        rb = ReplayBuffer(maxlen=int(args.buffer))
        print(f"Created new replay buffer (len={len(rb)} maxlen={rb.maxlen})")

    if args.clear_buffer:
        old_len = len(rb)
        rb = ReplayBuffer(maxlen=int(args.buffer))
        print(f"--clear_buffer: discarded {old_len} samples, fresh buffer")

    start_iter = 0
    if args.ckpt and os.path.exists(args.ckpt):
        try:
            start_iter = load_checkpoint(args.ckpt, net, opt, device=device, load_opt=args.resume_opt)
            print(f"Loaded checkpoint: {args.ckpt} (iter={start_iter})")
        except Exception as e:
            print(f"Could not load checkpoint ({e}).")

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
    pause_ev = mp.Event()  # unused in inline path, kept for eval gate API
    procs: List[BaseProcess] = []

    for it in range(start_iter + 1, args.iters + 1):
        t_iter0 = time.time()
        t_sp0 = time.time()
        c = init_counters()
        new_samples = 0
        games_collected = 0
        res_counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0, "other": 0}
        plies_sum = 0

        sf_engine = None
        if not args.no_sf:
            try:
                sf_engine = open_stockfish_engine(
                    stockfish_path="stockfish", threads=1, hash_mb=16,
                    elo=args.sf_elo, skill=None
                )
            except Exception:
                sf_engine = None
        try:
            sf_teacher_depth = None if int(args.sf_teacher_depth) <= 0 else int(args.sf_teacher_depth)
            for _ in trange(args.games_per_iter, desc=f"selfplay iter {it}",
                            leave=False, file=sys.stdout,
                            disable=not sys.stdout.isatty()):
                samples, res, plies, info = make_game_samples_unified(
                    net_white=net, net_black=None, device=device,
                    sims=args.sims, max_plies=args.max_plies,
                    resign_threshold=args.resign_threshold,
                    resign_patience=args.resign_patience,
                    sf_engine=sf_engine, sf_bootstrap_on_star=True,
                    sf_boot_time_ms=args.sf_boot_time_ms,
                    sf_cp_scale=args.sf_cp_scale, sf_cp_cap=args.sf_cp_cap,
                    sf_boot_prob=args.sf_boot_prob, sf_mate_cp=10000,
                    sf_boot_depth=args.sf_boot_depth,
                    mcts_value_mix=args.mcts_value_mix,
                    leaf_batch_size=max(1, int(args.mp_leaf_batch)),
                    sf_teacher_prob=args.sf_teacher_prob,
                    sf_teacher_mix=args.sf_teacher_mix,
                    sf_teacher_time_ms=args.sf_teacher_time_ms,
                    sf_teacher_depth=sf_teacher_depth,
                    sf_teacher_multipv=args.sf_teacher_multipv,
                    sf_teacher_cp_cap=args.sf_teacher_cp_cap,
                    sf_teacher_cp_soft_scale=args.sf_teacher_cp_soft_scale,
                    sf_teacher_eps=args.sf_teacher_eps,
                )
                accumulate_info(info, c, args)
                rb.add_game(samples)
                new_samples += len(samples)
                games_collected += 1
                plies_sum += int(plies)
                res_counts[res] = res_counts.get(res, 0) + 1
        finally:
            if sf_engine is not None:
                try:
                    sf_engine.quit()
                except Exception:
                    pass

        t_sp = time.time() - t_sp0
        avg_plies = plies_sum / max(1, games_collected)
        print(f"[iter {it}] selfplay_time={t_sp:.1f}s games={games_collected} buffer={len(rb)}")
        if games_collected:
            print_selfplay_info(it, c, games_collected)
        if len(rb) >= 1000:
            print_z_stats(it, rb)

        t_tr0 = time.time()
        net.train()
        last_m = None
        for step in range(args.steps_per_iter):
            if len(rb) < args.batch:
                continue
            batch_s = rb.sample_batch_mixed(
                args.batch, recent_frac=args.recent_frac,
                recent_window=args.recent_window,
                sharp_frac=args.sharp_frac,
                sharp_threshold=args.sharp_threshold,
            )
            batch = collate(batch_s, device)
            m = train_step(net, opt, batch, val_w=args.val_w, moves_left_w=args.moves_left_w)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*epoch parameter.*deprecated.*")
                scheduler.step()
            last_m = m
        t_tr = time.time() - t_tr0
        cur_lr = scheduler.get_last_lr()[0]
        t_iter = time.time() - t_iter0

        log_iteration_csv(
            args, it, c, games_collected, res_counts, avg_plies,
            new_samples, len(rb), last_m, t_sp, t_tr, t_iter, cur_lr
        )

        if args.save_every and (it % args.save_every == 0):
            save_iter(args, net, opt, it, rb, best_path)

        if args.eval_every and (it % args.eval_every == 0):
            _run_eval_gate(args, net, device, it, best_path, c, games_collected, len(rb),
                           pause_ev, procs)


def _run_eval_gate(args, net, device, it, best_path, c, games_collected, rb_len, pause_ev, procs):
    if args.workers > 0:
        pause_ev.set()
        time.sleep(2)

    old_nt = torch.get_num_threads()
    try:
        torch.set_num_threads(min(24, max(8, args.workers)))
        net.eval()
        print(f"Starting {args.eval_games} SF plays (elo={args.sf_eval_elo})")
        sf_score, sf_winrate, sf_elo_diff = play_vs_stockfish(
            net, device, stockfish_path=args.sf_path, sf_skill=args.sf_skill,
            sf_elo=args.sf_eval_elo, sf_movetime_ms=args.sf_movetime_ms,
            games=args.eval_games, sims=args.eval_sims, max_plies=args.sf_eval_max_plies
        )

        # Optional easy SF eval (using Skill Level, not Elo — since min UCI_Elo=1320)
        sf_easy_score = 0.0
        sf_easy_winrate = 0.0
        sf_easy_elo_diff = -2400.0
        if args.sf_eval_elo_easy >= 0:
            skill_lvl = args.sf_eval_elo_easy  # 0-20 range, 0=weakest (~1000 Elo)
            print(f"Starting {args.eval_games} SF plays (Skill Level {skill_lvl})")
            sf_easy_score, sf_easy_winrate, sf_easy_elo_diff = play_vs_stockfish(
                net, device, stockfish_path=args.sf_path, sf_skill=args.sf_skill,
                sf_elo=args.sf_eval_elo, sf_movetime_ms=args.sf_movetime_ms,
                games=args.eval_games, sims=args.eval_sims, max_plies=args.sf_eval_max_plies,
                use_skill_level=skill_lvl,
            )
            print(f"[iter {it}] eval vs SF Skill={skill_lvl}: "
                  f"score={sf_easy_score:.3f} winrate={sf_easy_winrate:.3f}")

        print(f"SF plays done, now making {args.rand_eval_games} random games")

        rnd_score, rnd_winrate, rnd_elo_diff = play_vs_random(
            net, device, games=args.rand_eval_games,
            sims=args.rand_eval_sims, max_plies=args.rand_max_plies,
        )

        best_metric_path = best_path + ".metric"
        best_kind_path = best_path + ".metric_kind"

        def _read_best_metric():
            try:
                with open(best_metric_path, "r") as f:
                    return float(f.read().strip())
            except Exception:
                return None

        def _write_best_metric(x: float):
            with open(best_metric_path, "w") as f:
                f.write(str(float(x)))

        def _read_best_kind():
            try:
                with open(best_kind_path, "r") as f:
                    return f.read().strip()
            except Exception:
                return ""

        def _write_best_kind(k: str):
            with open(best_kind_path, "w") as f:
                f.write(str(k))

        self_score = None
        self_winrate = None

        if os.path.exists(args.opponent_model):
            opp_eval = ChessNet().to(device)
            opp_eval.eval()
            try:
                opp_eval.load_state_dict(torch.load(args.opponent_model, map_location=device, weights_only=True))
            except Exception as e:
                print(f"[iter {it}] WARNING: opponent model incompatible, skipping self-eval ({e})")
                opp_eval = None
            if opp_eval is not None:
                print(f"Starting {args.self_eval_games} vs model plays")
                self_score, self_winrate = play_vs_model(
                    net, opp_eval, device,
                    games=args.self_eval_games,
                    sims=args.self_eval_sims,
                    max_plies=args.self_eval_max_plies,
                )
                print(f"[iter {it}] self-eval vs opponent: score={self_score:.3f} winrate={self_winrate:.3f}")

        SELF_TIE_EPS = 0.02
        SELF_MIN_WIN = 0.02
        metric_self = None

        if self_score is not None:
            sc = self_score
            wr = self_winrate if self_winrate is not None else 0.0
            if (abs(sc - 0.5) >= SELF_TIE_EPS) or (wr >= SELF_MIN_WIN):
                metric_self = sc

        if metric_self is not None:
            gate_kind = "self"
            metric = metric_self
        elif sf_score > 0.05:
            gate_kind = "sf_score"
            metric = float(sf_score)
        elif args.sf_eval_elo_easy >= 0 and sf_easy_score > 0.05:
            gate_kind = "sf_easy_score"
            metric = float(sf_easy_score)
        elif sf_winrate >= 0.02:
            gate_kind = "sf_win"
            metric = float(sf_winrate)
        elif args.sf_eval_elo_easy >= 0 and sf_easy_winrate >= 0.02:
            gate_kind = "sf_easy_win"
            metric = float(sf_easy_winrate)
        else:
            gate_kind = "rnd"
            metric = float(rnd_score)

        prev_kind = _read_best_kind()
        prev_metric = _read_best_metric()

        if (prev_metric is not None) and (not prev_kind):
            prev_kind = gate_kind
            _write_best_kind(prev_kind)

        promote = False
        reason = ""

        if prev_metric is None:
            if gate_kind == "self" and abs(metric - 0.5) < 0.02:
                promote = False
                reason = "init_but_self_tie"
            else:
                promote = True
                reason = "init"
        elif prev_kind != gate_kind:
            promote = False
            reason = f"kind_mismatch(prev={prev_kind}, now={gate_kind})"
        else:
            if metric >= prev_metric + args.gate_margin:
                promote = True
                reason = "improved"
            else:
                reason = "not_improved"

        if promote:
            tmp = best_path + f".tmp.{os.getpid()}"
            torch.save(net.state_dict(), tmp)
            os.replace(tmp, best_path)
            _write_best_metric(metric)
            _write_best_kind(gate_kind)
            print(f"[iter {it}] PROMOTED to best: {best_path} (metric={metric:.3f} prev={prev_metric} kind={gate_kind}")
        else:
            print(f"[iter {it}] not promoted (metric={metric:.3f} best={prev_metric} best_kind={prev_kind} now_kind={gate_kind}")

        best_kind_after = gate_kind if promote else prev_kind
        best_metric_after = metric if promote else prev_metric

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iter": it,
            "sf_elo": args.sf_eval_elo,
            "sf_movetime_ms": args.sf_movetime_ms,
            "sf_eval_games": args.eval_games,
            "sf_eval_sims": args.eval_sims,
            "sf_score": round(sf_score, 4),
            "sf_winrate": round(sf_winrate, 4),
            "sf_elo_diff_est": round(sf_elo_diff, 1),
            "sf_boot_rate": round(c["sf_boot_sum"] / max(1, games_collected), 4),
            "rnd_eval_games": args.rand_eval_games,
            "rnd_eval_sims": args.rand_eval_sims,
            "rnd_max_plies": args.rand_max_plies,
            "rnd_score": round(rnd_score, 4),
            "rnd_winrate": round(rnd_winrate, 4),
            "rnd_elo_diff_est": round(rnd_elo_diff, 1),
            "replay_len": rb_len,
            "self_score": round(self_score, 4) if self_score is not None else "",
            "self_winrate": round(self_winrate, 4) if self_winrate is not None else "",
            "gate_kind": gate_kind,
            "gate_metric": round(float(metric), 6),
            "best_kind": prev_kind,
            "best_metric": round(float(prev_metric), 6) if prev_metric is not None else "",
            "gate_margin": args.gate_margin,
            "promoted": int(bool(promote)),
            "promote_reason": reason,
            "best_kind_after": best_kind_after,
            "best_metric_after": round(float(best_metric_after), 6) if best_metric_after is not None else "",
            "sf_cp_scale_used": args.sf_cp_scale,
            "sf_cp_cap_used": args.sf_cp_cap,
            "forced_end_rate": round(c["forced_end_sum"] / max(1, games_collected), 4),
            "forced_resign_rate": round(c["forced_resign_sum"] / max(1, games_collected), 4),
            "forced_draw_rate": round(c["forced_draw_sum"] / max(1, games_collected), 4),
            "sf_fail_sum": c["sf_fail_sum"],
            "sf_easy_skill": args.sf_eval_elo_easy if args.sf_eval_elo_easy >= 0 else "",
            "sf_easy_score": round(sf_easy_score, 4) if args.sf_eval_elo_easy >= 0 else "",
            "sf_easy_winrate": round(sf_easy_winrate, 4) if args.sf_eval_elo_easy >= 0 else "",
        }
        append_eval_csv(args.eval_csv, row)

        easy_str = ""
        if args.sf_eval_elo_easy >= 0:
            easy_str = f" | vs SF Skill={args.sf_eval_elo_easy}: score={sf_easy_score:.3f}"
        print(
            f"[iter {it}] eval vs SF elo={args.sf_eval_elo}: score={sf_score:.3f}, "
            f"elo_diff={sf_elo_diff:+.0f} | "
            f"vs RANDOM: score={rnd_score:.3f}, elo_diff={rnd_elo_diff:+.0f}"
            f"{easy_str} "
            f"(logged to {args.eval_csv})"
        )
        net.train()
    finally:
        torch.set_num_threads(old_nt)
        if args.workers > 0:
            pause_ev.clear()
