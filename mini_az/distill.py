"""Supervised distillation from Stockfish — NNUE-style training without MCTS.

Generates positions by playing random/semi-random games, evaluates them with
Stockfish (depth search for value, multipv for policy), and creates training
samples directly. ~100-1000× faster than RL/MCTS for bootstrapping a network.
"""

import os
import random
import time
import traceback
import multiprocessing as mp

import numpy as np
import chess
import torch

from .config import print, apply_random_opening, material_score
from .encoding import (
    board_to_tensor, legal_moves_canonical, HISTORY_STEPS,
)
from .training import Sample, ReplayBuffer, collate, train_step
from .stockfish import (
    open_stockfish_engine, sf_eval_cp_white, cp_to_z,
    sf_teacher_policy_legal,
)
from .network import ChessNet


def _generate_random_position(max_random_plies: int = 120) -> tuple:
    """Play random moves to reach a diverse position. Return (board, history)."""
    board = chess.Board()
    history: list[chess.Board] = []

    # Start from opening book ~30% of the time for realistic positions
    if random.random() < 0.3:
        apply_random_opening(board, max_book_plies=8)

    n_plies = random.randint(4, max_random_plies)
    for _ in range(n_plies):
        if board.is_game_over():
            break
        legals = list(board.legal_moves)
        if not legals:
            break
        # Semi-random: 70% uniform random, 30% capture/check bias
        if random.random() < 0.3:
            captures = [m for m in legals if board.is_capture(m)]
            checks = [m for m in legals if board.gives_check(m)]
            biased = captures + checks
            if biased:
                mv = random.choice(biased)
            else:
                mv = random.choice(legals)
        else:
            mv = random.choice(legals)
        history = [board.copy()] + history[:HISTORY_STEPS - 1]
        board.push(mv)

    return board, history


def _make_distill_sample(
    sf_engine,
    board: chess.Board,
    history: list,
    sf_eval_depth: int = 12,
    sf_policy_depth: int = 10,
    sf_multipv: int = 8,
    sf_cp_scale: float = 600.0,
    sf_cp_cap: int = 1200,
    sf_policy_cp_cap: int = 800,
    sf_policy_cp_soft_scale: float = 150.0,
) -> Sample | None:
    """Create a training sample from SF evaluation of a single position."""
    if board.is_game_over():
        return None

    legals_canon = legal_moves_canonical(board)
    if not legals_canon:
        return None
    legal_moves = [m[3] for m in legals_canon]

    # 1. SF value (depth search)
    cp = sf_eval_cp_white(
        sf_engine, board, depth=sf_eval_depth, mate_cp=10000
    )
    if cp is None:
        return None

    # Convert to side-to-move perspective
    cp_stm = cp if board.turn == chess.WHITE else -cp
    z = cp_to_z(cp_stm, cp_scale=sf_cp_scale, cp_cap=sf_cp_cap)

    # WDL from z
    if z > 0:
        wdl = np.array([z, 1.0 - z, 0.0], dtype=np.float32)
    elif z < 0:
        wdl = np.array([0.0, 1.0 + z, -z], dtype=np.float32)
    else:
        wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 2. SF policy (multipv)
    pi = sf_teacher_policy_legal(
        sf_engine, board, legal_moves,
        depth=sf_policy_depth,
        multipv=sf_multipv,
        mate_cp=10000,
        cp_cap=sf_policy_cp_cap,
        cp_soft_scale=sf_policy_cp_soft_scale,
        eps=0.01,
    )
    if pi is None:
        # Fallback: uniform policy
        pi = np.ones(len(legal_moves), dtype=np.float32) / len(legal_moves)

    # 3. Encode board
    bt = board_to_tensor(board, history=history[:HISTORY_STEPS]).numpy()
    fs = np.array([m[0] for m in legals_canon], dtype=np.int64)
    ts = np.array([m[1] for m in legals_canon], dtype=np.int64)
    pr = np.array([m[2] for m in legals_canon], dtype=np.int64)

    # Estimate plies left from material/phase
    total_pieces = len(board.piece_map())
    plies_left = float(max(10, min(120, total_pieces * 4)))

    return Sample(bt, fs, ts, pr, pi.astype(np.float32), float(z), wdl, plies_left)


def _distill_worker(
    worker_id: int,
    out_q: mp.Queue,
    stop_ev,
    sf_path: str,
    sf_elo: int,
    sf_eval_depth: int,
    sf_policy_depth: int,
    sf_multipv: int,
    sf_cp_scale: float,
    sf_cp_cap: int,
    positions_per_game: int,
    max_random_plies: int,
):
    """Worker process: generate positions and SF-evaluate them."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    try:
        sf_engine = open_stockfish_engine(
            sf_path, threads=1, hash_mb=16, elo=None, skill=None
        )
    except Exception as e:
        print(f"[distill worker {worker_id}] SF failed: {e}")
        return

    samples_made = 0
    try:
        while not stop_ev.is_set():
            # Generate a random game and sample positions from it
            board = chess.Board()
            history: list[chess.Board] = []

            if random.random() < 0.3:
                apply_random_opening(board, max_book_plies=8)

            game_positions: list[tuple] = []
            n_plies = random.randint(8, max_random_plies)
            for ply in range(n_plies):
                if board.is_game_over():
                    break
                legals = list(board.legal_moves)
                if not legals:
                    break

                # Save position for potential sampling
                if ply >= 4:  # skip first few plies (too trivial)
                    game_positions.append((board.copy(), list(history)))

                # Semi-random play
                if random.random() < 0.25:
                    captures = [m for m in legals if board.is_capture(m)]
                    checks = [m for m in legals if board.gives_check(m)]
                    biased = captures + checks
                    mv = random.choice(biased) if biased else random.choice(legals)
                else:
                    mv = random.choice(legals)

                history = [board.copy()] + history[:HISTORY_STEPS - 1]
                board.push(mv)

            # Sample N positions from the game
            if game_positions:
                k = min(positions_per_game, len(game_positions))
                chosen = random.sample(game_positions, k)

                batch_samples = []
                for pos_board, pos_hist in chosen:
                    s = _make_distill_sample(
                        sf_engine, pos_board, pos_hist,
                        sf_eval_depth=sf_eval_depth,
                        sf_policy_depth=sf_policy_depth,
                        sf_multipv=sf_multipv,
                        sf_cp_scale=sf_cp_scale,
                        sf_cp_cap=sf_cp_cap,
                    )
                    if s is not None:
                        batch_samples.append(s)

                if batch_samples:
                    try:
                        out_q.put(batch_samples, timeout=5)
                        samples_made += len(batch_samples)
                    except Exception:
                        pass

    except Exception:
        traceback.print_exc()
    finally:
        try:
            sf_engine.quit()
        except Exception:
            pass
    print(f"[distill worker {worker_id}] done, made {samples_made} samples")


def run_distillation(
    net: ChessNet,
    device: str,
    sf_path: str = "stockfish",
    sf_elo: int = 2000,
    workers: int = 80,
    target_samples: int = 500_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    epochs: int = 3,
    sf_eval_depth: int = 12,
    sf_policy_depth: int = 10,
    sf_multipv: int = 8,
    sf_cp_scale: float = 600.0,
    sf_cp_cap: int = 1200,
    positions_per_game: int = 10,
    max_random_plies: int = 120,
    val_w: float = 2.0,
    moves_left_w: float = 0.15,
    save_path: str = "mini_az.pt",
    save_every_k: int = 50_000,
):
    """Generate SF-labeled positions and train supervised."""
    print(f"=== DISTILLATION MODE ===")
    print(f"Target: {target_samples} samples, {workers} workers")
    print(f"SF eval depth={sf_eval_depth}, policy depth={sf_policy_depth}, multipv={sf_multipv}")

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=2048)
    stop_ev = ctx.Event()

    procs = []
    for wid in range(workers):
        p = ctx.Process(
            target=_distill_worker,
            args=(wid, out_q, stop_ev, sf_path, sf_elo,
                  sf_eval_depth, sf_policy_depth, sf_multipv,
                  sf_cp_scale, sf_cp_cap, positions_per_game,
                  max_random_plies),
            daemon=True,
        )
        p.start()
        procs.append(p)

    print(f"Started {workers} distillation workers")

    # Phase 1: Collect samples
    rb = ReplayBuffer(maxlen=target_samples + 100_000)
    t0 = time.time()
    total_collected = 0

    try:
        while total_collected < target_samples:
            try:
                batch_samples = out_q.get(timeout=30)
            except Exception:
                alive = sum(1 for p in procs if p.is_alive())
                if alive == 0:
                    print("All workers died!")
                    break
                print(f"  Waiting... collected={total_collected} alive={alive}")
                continue

            rb.add_game(batch_samples)
            total_collected += len(batch_samples)

            if total_collected % 5000 < len(batch_samples):
                elapsed = time.time() - t0
                rate = total_collected / elapsed
                eta = (target_samples - total_collected) / max(1, rate)
                print(f"  Collected {total_collected}/{target_samples} "
                      f"({rate:.0f} samples/s, ETA {eta:.0f}s)")

    finally:
        stop_ev.set()
        time.sleep(1)
        for p in procs:
            p.join(timeout=2)
        for p in procs:
            if p.is_alive():
                p.kill()

    elapsed = time.time() - t0
    print(f"Collection done: {total_collected} samples in {elapsed:.0f}s "
          f"({total_collected/elapsed:.0f}/s)")

    if total_collected < batch_size:
        print("Too few samples, aborting distillation")
        return

    # Phase 2: Train supervised
    print(f"\n=== TRAINING PHASE: {epochs} epochs, batch={batch_size} ===")
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = max(1, len(rb) // batch_size)
    total_steps = epochs * steps_per_epoch
    warmup = min(200, total_steps // 10)
    cosine_steps = total_steps - warmup

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=warmup
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cosine_steps), eta_min=lr * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
    )
    print(f"LR: warmup {warmup} steps, cosine {cosine_steps}, total {total_steps}")

    global_step = 0
    t_train0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_pol = 0.0
        epoch_val = 0.0
        epoch_steps = 0

        for step in range(steps_per_epoch):
            batch_s = rb.sample_batch(batch_size)
            batch = collate(batch_s, device)
            m = train_step(net, opt, batch, val_w=val_w, moves_left_w=moves_left_w)
            scheduler.step()
            global_step += 1
            epoch_steps += 1

            epoch_loss += m["loss"]
            epoch_pol += m["pol"]
            epoch_val += m["val"]

            if global_step % 100 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  [epoch {epoch+1}/{epochs}] step {global_step}/{total_steps} "
                      f"loss={m['loss']:.4f} pol={m['pol']:.4f} val={m['val']:.4f} "
                      f"vz_corr={m['vz_corr']:+.3f} lr={cur_lr:.2e}")

            # Save periodically
            if save_every_k > 0 and global_step % (save_every_k // batch_size) == 0:
                torch.save(net.state_dict(), save_path)

        avg_loss = epoch_loss / max(1, epoch_steps)
        avg_pol = epoch_pol / max(1, epoch_steps)
        avg_val = epoch_val / max(1, epoch_steps)
        t_ep = time.time() - t_train0
        print(f"  Epoch {epoch+1}/{epochs} done: "
              f"avg_loss={avg_loss:.4f} pol={avg_pol:.4f} val={avg_val:.4f} "
              f"time={t_ep:.0f}s")

    # Final save
    torch.save(net.state_dict(), save_path)
    t_total = time.time() - t_train0
    print(f"Distillation complete: {global_step} steps in {t_total:.0f}s")
    print(f"Model saved to {save_path}")
