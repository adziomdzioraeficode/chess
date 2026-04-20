"""Self-play game generation and multiprocessing worker."""

import os
import random
import time
import queue
import traceback

import numpy as np
import chess
import chess.engine
import torch

from typing import TypedDict

from .config import (
    print, NO_PROGRESS_HALFMOVE, apply_random_opening, material_score
)
from .encoding import board_to_tensor, legal_moves_canonical, HISTORY_STEPS
from .network import ChessNet
from .mcts import mcts_search
from .training import Sample
from .stockfish import (
    open_stockfish_engine, sf_eval_cp_white, cp_to_z, sf_teacher_policy_legal
)


class _SfCommon(TypedDict):
    sf_engine: chess.engine.SimpleEngine | None
    sf_boot_time_ms: int
    sf_cp_scale: float
    sf_boot_prob: float
    sf_cp_cap: int
    sf_boot_depth: int | None
    mcts_value_mix: float
    sf_teacher_prob: float
    sf_teacher_mix: float
    sf_teacher_time_ms: int
    sf_teacher_depth: int | None
    sf_teacher_multipv: int
    sf_teacher_cp_cap: int
    sf_teacher_cp_soft_scale: float
    sf_teacher_eps: float


def default_weights_shm_path() -> str:
    """Path under tmpfs if available, otherwise $TMPDIR — one file shared
    across all workers, updated atomically via tmp+rename."""
    base = "/dev/shm" if os.path.isdir("/dev/shm") else None
    if base is None:
        import tempfile
        base = tempfile.gettempdir()
    return os.path.join(base, f"mini_az_weights_{os.getpid()}.pt")


def broadcast_weights(weights_path: str, net: ChessNet, version_value):
    """Publish *net* to the shared weights file and bump the version counter.

    All self-play workers memory-map the same file; the version counter is a
    single 64-bit integer they poll each loop iteration. This replaces the
    old ~22 MB x N_workers pickle-into-Queue broadcast (which spent ~0.5 s
    and ~2 GB of RSS per iteration).
    """
    net_cpu_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    tmp = f"{weights_path}.tmp.{os.getpid()}"
    torch.save(net_cpu_sd, tmp)
    os.replace(tmp, weights_path)  # atomic on POSIX; workers never see a torn file
    with version_value.get_lock():
        version_value.value += 1


# Backward-compatible alias: the initial broadcast is identical to any other
# broadcast — workers pick it up by polling the version counter.
broadcast_weights_initial = broadcast_weights


def _load_weights_shm(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


def make_game_samples_unified(
    net_white: ChessNet,
    device: str,
    sims: int,
    sf_boot_time_ms: int,
    sf_cp_scale: float,
    sf_cp_cap: int,
    sf_boot_prob: float,
    net_black: ChessNet | None = None,
    train_only_color: bool | None = None,
    max_plies: int = 250,
    resign_threshold: float = -0.995,
    resign_patience: int = 20,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    claim_draw: bool = False,
    opening_random_plies: int = 8,
    material_adjudicate: int = 5,
    mcts_value_mix: float = 0.5,
    sf_mate_cp: int = 10000,
    sf_engine: chess.engine.SimpleEngine | None = None,
    sf_bootstrap_on_star: bool = False,
    sf_boot_depth: int | None = None,
    sf_teacher_prob: float = 0.50,
    sf_teacher_mix: float = 0.50,
    sf_teacher_time_ms: int = 15,
    sf_teacher_depth: int | None = None,
    sf_teacher_multipv: int = 4,
    sf_teacher_cp_cap: int = 800,
    sf_teacher_cp_soft_scale: float = 120.0,
    sf_teacher_eps: float = 0.01,
    leaf_batch_size: int = 1,
):
    net_black = net_white if net_black is None else net_black

    forced_res = None
    forced_kind = ""
    board = chess.Board()
    played_book = apply_random_opening(board, opening_random_plies)
    ply0 = board.ply()
    trajectory = []
    bad_streak = 0
    pi_ents, v_list = [], []
    threefold_penalized = 0
    draw_claim_streak = 0
    rep_plies = 0
    actual_rep_plies = 0
    rep_moves = 0
    # Maintain position history for encoding (most recent first)
    board_history: list[chess.Board] = []

    while (not board.is_game_over(claim_draw=claim_draw)) and (board.ply() - ply0) < max_plies:
        net_to_move = net_white if board.turn == chess.WHITE else net_black

        tply = board.ply() - ply0
        is_pure_selfplay = (net_black is net_white)
        DIR_PLY = 30
        da = dirichlet_alpha if (is_pure_selfplay and tply < DIR_PLY) else 0.0
        de = dirichlet_eps   if (is_pure_selfplay and tply < DIR_PLY) else 0.0

        # Pass board history to MCTS for history-aware encoding
        visits, v_now = mcts_search(
            net_to_move, board, device, sims=sims,
            policy_temp=1.0,
            dirichlet_alpha=da, dirichlet_eps=de,
            history=board_history[:HISTORY_STEPS],
            leaf_batch_size=leaf_batch_size,
        )

        ms_now = material_score(board)
        adv = ms_now if board.turn == chess.WHITE else -ms_now

        drawish_now = (board.is_repetition(2)
               or board.can_claim_fifty_moves()
               or board.halfmove_clock >= NO_PROGRESS_HALFMOVE)

        draw_claim_min_ply = max(60, int(0.55 * max_plies))
        if drawish_now and adv < 2 and (board.ply() - ply0) > draw_claim_min_ply and abs(v_now) < 0.15:
            draw_claim_streak += 1
        else:
            draw_claim_streak = 0
        if draw_claim_streak >= 3:
            forced_res = "1/2-1/2"
            forced_kind = "draw_claim"
            break

        RESIGN_MIN_PLY = min(80, max(30, max_plies // 3))
        if tply >= RESIGN_MIN_PLY:
            if v_now < resign_threshold:
                bad_streak += 1
                if bad_streak == resign_patience:
                    print(f"[resign] tply={tply} turn={'W' if board.turn else 'B'} v_now={v_now:.3f} halfmove={board.halfmove_clock}")
            else:
                bad_streak = 0
            if bad_streak >= resign_patience:
                forced_res = "0-1" if board.turn == chess.WHITE else "1-0"
                forced_kind = "resign"
                break
        else:
            bad_streak = 0

        if not visits:
            break

        TEMP_PLY_FULL = 12
        if tply < TEMP_PLY_FULL:
            temperature = 1.0
        elif tply < 28:
            temperature = 0.35
        else:
            temperature = 0.08

        legals = legal_moves_canonical(board)
        legal_real = [m[3] for m in legals]

        counts_raw = np.array([visits.get(mv, 0) for mv in legal_real], dtype=np.float64)
        if counts_raw.sum() <= 0:
            counts_raw += 1.0
        pi_target = counts_raw / (counts_raw.sum() + 1e-12)

        use_teacher = (
            (sf_engine is not None)
            and (sf_teacher_prob > 0.0)
            and (random.random() < sf_teacher_prob)
            and (tply < 160)
            and (not drawish_now)
        )

        if use_teacher:
            teacher_p = sf_teacher_policy_legal(
                sf_engine, board, legal_real,
                movetime_ms=sf_teacher_time_ms,
                depth=sf_teacher_depth,
                multipv=sf_teacher_multipv,
                mate_cp=sf_mate_cp,
                cp_cap=sf_teacher_cp_cap,
                cp_soft_scale=sf_teacher_cp_soft_scale,
                eps=sf_teacher_eps,
            )
            if teacher_p is not None:
                alpha = float(np.clip(sf_teacher_mix, 0.0, 1.0))
                pi_target = ((1.0 - alpha) * pi_target + alpha * teacher_p).astype(np.float32)
                pi_target = pi_target / (pi_target.sum() + 1e-12)

        counts_beh = counts_raw.copy()

        if adv >= 3:
            rep_penalty, fifty_penalty, pat_penalty = 0.10, 0.15, 0.15
        elif adv >= 1:
            rep_penalty, fifty_penalty, pat_penalty = 0.25, 0.30, 0.30
        else:
            rep_penalty, fifty_penalty, pat_penalty = 0.50, 0.55, 0.50

        rep_this_ply = False
        for i, mv in enumerate(legal_real):
            b2 = board.copy()
            b2.push(mv)
            if b2.is_stalemate():
                counts_beh[i] *= pat_penalty
                continue
            if b2.can_claim_threefold_repetition():
                counts_beh[i] *= rep_penalty
                rep_moves += 1
                rep_this_ply = True
                threefold_penalized += 1
            if b2.can_claim_fifty_moves() or b2.halfmove_clock >= NO_PROGRESS_HALFMOVE:
                counts_beh[i] *= fifty_penalty

        if rep_this_ply:
            rep_plies += 1
        counts_beh = counts_beh + 1e-6 * np.random.random(size=len(counts_beh))

        s_beh = counts_beh.sum()
        if s_beh <= 0:
            counts_beh = counts_raw.copy()
            s_beh = counts_beh.sum()
        pi_beh = counts_beh / (s_beh + 1e-12)

        OPEN_PLY = 16
        OPEN_TEMP = 1.25
        eff_temp = temperature
        if tply < (played_book + OPEN_PLY):
            eff_temp = max(eff_temp, OPEN_TEMP)

        if eff_temp < 1e-3:
            idx = int(np.argmax(pi_beh))
        else:
            if abs(eff_temp - 1.0) < 1e-6:
                p = pi_beh
            else:
                p = np.power(pi_beh, 1.0 / eff_temp)
                ps = p.sum()
                p = (p / ps) if ps > 0 else pi_beh
            idx = int(np.random.choice(len(p), p=p))

        eps = 1e-12
        pi_ents.append(float(-(pi_target * np.log(pi_target + eps)).sum()))
        v_list.append(float(v_now))

        bt = board_to_tensor(board, history=board_history[:HISTORY_STEPS]).numpy()
        fs = np.array([m[0] for m in legals], dtype=np.int64)
        ts = np.array([m[1] for m in legals], dtype=np.int64)
        pr = np.array([m[2] for m in legals], dtype=np.int64)
        if train_only_color is None or board.turn == train_only_color:
            trajectory.append((bt, fs, ts, pr, pi_target.astype(np.float32), board.turn, float(v_now), int(tply)))

        chosen_mv = legal_real[idx]
        # Update history: prepend current board before pushing move
        board_history = [board.copy()] + board_history[:HISTORY_STEPS - 1]
        board.push(chosen_mv)

        if board.can_claim_threefold_repetition() or board.is_repetition(2):
            actual_rep_plies += 1

    res = forced_res if forced_res is not None else board.result(claim_draw=claim_draw)

    z_boot_white: float | None = None
    sf_cp: int | None = None
    sf_fail: int = 0
    ms = material_score(board)

    if res == "1/2-1/2" and forced_res is not None:
        if ms >= material_adjudicate:
            res = "1-0"
        elif ms <= -material_adjudicate:
            res = "0-1"

    ended_star = (res == "*")
    if ended_star:
        ms = material_score(board)
        if ms >= material_adjudicate:
            res = "1-0"
        elif ms <= -material_adjudicate:
            res = "0-1"
        else:
            res = "1/2-1/2"

        if (sf_bootstrap_on_star
            and sf_engine is not None
            and (random.random() < sf_boot_prob)):
            cp = sf_eval_cp_white(
                sf_engine, board,
                movetime_ms=sf_boot_time_ms,
                depth=sf_boot_depth,
                mate_cp=sf_mate_cp
            )
            if cp is not None:
                sf_cp = cp
                z_boot_white = cp_to_z(sf_cp, cp_scale=sf_cp_scale, cp_cap=sf_cp_cap)
            else:
                sf_fail = 1

    winner = chess.WHITE if res == "1-0" else chess.BLACK if res == "0-1" else None

    if (winner is None) and (z_boot_white is None) and (sf_engine is not None) and (random.random() < sf_boot_prob):
        cp = sf_eval_cp_white(
            sf_engine, board,
            movetime_ms=sf_boot_time_ms,
            depth=sf_boot_depth,
            mate_cp=sf_mate_cp
        )
        if cp is not None:
            sf_cp = cp
            z_boot_white = cp_to_z(sf_cp, cp_scale=sf_cp_scale, cp_cap=sf_cp_cap)
        else:
            sf_fail = 1

    z_draw_white = 0.0  # Clean draw; SF bootstrap overrides if available

    if winner == chess.WHITE:
        z_white_end = 1.0
    elif winner == chess.BLACK:
        z_white_end = -1.0
    else:
        z_white_end = z_boot_white if z_boot_white is not None else z_draw_white

    samples = []
    lam = float(np.clip(mcts_value_mix, 0.0, 1.0))
    final_tply = max(0, board.ply() - ply0)
    for bt, fs, ts, pr, pi, turn, v_mcts, sample_tply in trajectory:
        z_game = z_white_end if turn == chess.WHITE else -z_white_end
        z_target = (1.0 - lam) * float(z_game) + lam * float(v_mcts)
        z_target = float(np.clip(z_target, -1.0, 1.0))
        # Compute WDL target from z_target:
        # z in [-1, 1] → map to WDL probabilities
        # z=1 → [1,0,0] (win), z=-1 → [0,0,1] (loss), z=0 → [0,1,0] (draw)
        # For fractional z (SF bootstrap), interpolate smoothly
        if z_target > 0:
            wdl = np.array([z_target, 1.0 - z_target, 0.0], dtype=np.float32)
        elif z_target < 0:
            wdl = np.array([0.0, 1.0 + z_target, -z_target], dtype=np.float32)
        else:
            wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        plies_left = float(max(0, final_tply - int(sample_tply)))
        samples.append(Sample(bt, fs, ts, pr, pi.astype(np.float32), z_target, wdl, plies_left))

    info = {
        "avg_pi_ent": float(np.mean(pi_ents)) if pi_ents else 0.0,
        "avg_v": float(np.mean(v_list)) if v_list else 0.0,
        "min_v": float(np.min(v_list)) if v_list else 0.0,
        "max_v": float(np.max(v_list)) if v_list else 0.0,
        "threefold_penalized": int(threefold_penalized),
        "opening_book_plies": int(played_book),
        "forced_res_str": forced_res if forced_res is not None else "",
        "ended_star": bool(ended_star),
        "sf_boot_used": bool(z_boot_white is not None),
        "sf_available": bool(sf_engine is not None),
        "sf_cp_white": int(sf_cp) if sf_cp is not None else 0,
        "sf_z_white": float(z_boot_white) if z_boot_white is not None else 0.0,
        "rep_moves_penalized": int(rep_moves),
        "rep_plies_with_rep": int(rep_plies),
        "rep_plies_actual": int(actual_rep_plies),
        "sf_boot_src": "forced_draw" if (forced_res is not None and z_boot_white is not None and not ended_star) else ("star" if ended_star and z_boot_white is not None else ""),
        "forced_end": forced_res is not None,
        "forced_kind": forced_kind,
        "sf_fail": sf_fail
    }
    sf_cp_used = 0
    if sf_cp is not None:
        cap = int(sf_cp_cap) if sf_cp_cap is not None else int(sf_cp)
        sf_cp_used = int(max(-cap, min(cap, int(sf_cp))))
    info["sf_cp_used"] = int(sf_cp_used)
    info["sf_cp_scale"] = float(sf_cp_scale)
    info["sf_cp_cap"] = int(sf_cp_cap)
    return samples, res, board.ply(), info


def _run_game_vs(net, opponent_net, device, sims, max_plies, resign_threshold,
                 resign_patience, sf_engine, sf_boot_time_ms, sf_cp_scale,
                 sf_boot_prob, sf_cp_cap, sf_boot_depth, mcts_value_mix,
                 sf_teacher_prob, sf_teacher_mix, sf_teacher_time_ms,
                 sf_teacher_depth, sf_teacher_multipv, sf_teacher_cp_cap,
                 sf_teacher_cp_soft_scale, sf_teacher_eps, train_color,
                 leaf_batch_size: int = 1):
    """Helper: run one game net vs opponent_net, training only train_color's moves."""
    return make_game_samples_unified(
        net_white=net if train_color == chess.WHITE else opponent_net,
        net_black=opponent_net if train_color == chess.WHITE else net,
        device=device, sims=sims,
        max_plies=max_plies, resign_threshold=resign_threshold,
        resign_patience=resign_patience,
        sf_engine=sf_engine, sf_bootstrap_on_star=True,
        sf_boot_time_ms=sf_boot_time_ms, sf_cp_scale=sf_cp_scale,
        sf_boot_prob=sf_boot_prob, sf_cp_cap=sf_cp_cap,
        sf_mate_cp=10000, sf_boot_depth=sf_boot_depth,
        mcts_value_mix=mcts_value_mix,
        sf_teacher_prob=sf_teacher_prob, sf_teacher_mix=sf_teacher_mix,
        sf_teacher_time_ms=sf_teacher_time_ms, sf_teacher_depth=sf_teacher_depth,
        sf_teacher_multipv=sf_teacher_multipv, sf_teacher_cp_cap=sf_teacher_cp_cap,
        sf_teacher_cp_soft_scale=sf_teacher_cp_soft_scale,
        sf_teacher_eps=sf_teacher_eps,
        train_only_color=train_color,
        leaf_batch_size=leaf_batch_size,
    )


def selfplay_worker(
    worker_id: int,
    weights_path: str,
    weights_version,
    out_q,
    stop_ev,
    pause_ev,
    sims: int,
    max_plies: int,
    resign_threshold: float,
    resign_patience: int,
    best_path: str,
    mix_best: float,
    best_reload_sec: float,
    opp_path: str,
    mix_opp: float,
    opp_reload_sec: float,
    sf_boot_time_ms: int,
    sf_boot_prob: float,
    sf_cp_scale: float,
    sf_cp_cap: int,
    sf_boot_depth: int,
    sf_teacher_prob: float,
    sf_teacher_mix: float,
    sf_teacher_time_ms: int,
    sf_teacher_depth: int | None,
    sf_teacher_multipv: int,
    sf_teacher_cp_cap: int,
    sf_teacher_cp_soft_scale: float,
    sf_teacher_eps: float,
    enable_sf: bool = False,
    sf_elo: int = 1320,
    mcts_value_mix: float = 0.5,
    leaf_batch_size: int = 1,
    use_bf16_inference: bool = False,
):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    device = "cpu"

    sf_engine = None
    if enable_sf:
        try:
            sf_engine = open_stockfish_engine(
                stockfish_path="stockfish", threads=1, hash_mb=16,
                elo=sf_elo, skill=None
            )
            print(f"[worker {worker_id}] Stockfish OK")
        except Exception as e:
            print(f"[worker {worker_id}] stockfish init failed: {e}")
            sf_engine = None

    try:
        net = ChessNet().to(device)
        net.use_bf16_inference = bool(use_bf16_inference)
        net.eval()

        best_net = ChessNet().to(device)
        best_net.use_bf16_inference = bool(use_bf16_inference)
        best_net.eval()
        best_mtime = 0.0
        last_check = 0.0

        try:
            if os.path.exists(best_path):
                sd = torch.load(best_path, map_location="cpu", weights_only=True)
                best_net.load_state_dict(sd)
                best_mtime = os.stat(best_path).st_mtime
        except Exception:
            pass

        def maybe_reload_best():
            nonlocal best_mtime, last_check
            now = time.time()
            if now - last_check < best_reload_sec:
                return
            last_check = now
            try:
                st = os.stat(best_path)
                if st.st_mtime > best_mtime:
                    sd = torch.load(best_path, map_location="cpu", weights_only=True)
                    best_net.load_state_dict(sd)
                    best_mtime = st.st_mtime
            except Exception:
                pass

        opp_net = ChessNet().to(device)
        opp_net.use_bf16_inference = bool(use_bf16_inference)
        opp_net.eval()
        opp_mtime = 0.0
        opp_last_check = 0.0

        try:
            if os.path.exists(opp_path):
                sd = torch.load(opp_path, map_location="cpu", weights_only=True)
                opp_net.load_state_dict(sd)
                opp_mtime = os.stat(opp_path).st_mtime
        except Exception:
            pass

        def maybe_reload_opp():
            nonlocal opp_mtime, opp_last_check
            now = time.time()
            if now - opp_last_check < opp_reload_sec:
                return
            opp_last_check = now
            try:
                st = os.stat(opp_path)
                if st.st_mtime > opp_mtime:
                    sd = torch.load(opp_path, map_location="cpu", weights_only=True)
                    opp_net.load_state_dict(sd)
                    opp_mtime = st.st_mtime
            except Exception:
                pass

        # Wait for initial weights — main publishes them via shared file and
        # bumps `weights_version`. We poll the counter cheaply (no locks needed
        # for reads of an `mp.Value('q')`).
        last_seen_version = 0
        while True:
            v = int(weights_version.value)
            if v > 0:
                try:
                    sd = _load_weights_shm(weights_path, map_location="cpu")
                    net.load_state_dict(sd)
                    last_seen_version = v
                    if worker_id == 0:
                        print(f"[worker {worker_id}] got weights (version={v})")
                    break
                except Exception as e:
                    # File may be momentarily missing between tmp+rename; retry.
                    if stop_ev.is_set():
                        return
                    print(f"[worker {worker_id}] initial weight load retry: {e}")
                    time.sleep(0.2)
                    continue
            if stop_ev.is_set():
                return
            time.sleep(0.1)

        sf_common = _SfCommon(
            sf_engine=sf_engine,
            sf_boot_time_ms=sf_boot_time_ms, sf_cp_scale=sf_cp_scale,
            sf_boot_prob=sf_boot_prob, sf_cp_cap=sf_cp_cap,
            sf_boot_depth=sf_boot_depth, mcts_value_mix=mcts_value_mix,
            sf_teacher_prob=sf_teacher_prob, sf_teacher_mix=sf_teacher_mix,
            sf_teacher_time_ms=sf_teacher_time_ms, sf_teacher_depth=sf_teacher_depth,
            sf_teacher_multipv=sf_teacher_multipv, sf_teacher_cp_cap=sf_teacher_cp_cap,
            sf_teacher_cp_soft_scale=sf_teacher_cp_soft_scale,
            sf_teacher_eps=sf_teacher_eps,
        )

        while not stop_ev.is_set():
            if pause_ev.is_set():
                time.sleep(1)
                continue

            # Reload network weights if a new version was published.
            v = int(weights_version.value)
            if v != last_seen_version:
                try:
                    sd = _load_weights_shm(weights_path, map_location="cpu")
                    net.load_state_dict(sd)
                    last_seen_version = v
                except Exception as e:
                    # Transient (tmp+rename) — try again on next loop.
                    print(f"[worker {worker_id}] weight reload retry: {e}")

            maybe_reload_best()
            maybe_reload_opp()

            r = random.random()
            use_best = (r < mix_best) and os.path.exists(best_path)
            use_opp  = (not use_best) and (r < mix_best + mix_opp) and os.path.exists(opp_path)

            if use_best:
                train_color = chess.WHITE if random.random() < 0.5 else chess.BLACK
                game_samples, res, plies, info = _run_game_vs(
                    net, best_net, device, sims, max_plies,
                    resign_threshold, resign_patience, train_color=train_color,
                    leaf_batch_size=leaf_batch_size,
                    **sf_common
                )
                info["vs_best"] = True
                info["vs_opp"] = False
            elif use_opp:
                train_color = chess.WHITE if random.random() < 0.5 else chess.BLACK
                game_samples, res, plies, info = _run_game_vs(
                    net, opp_net, device, sims, max_plies,
                    resign_threshold, resign_patience, train_color=train_color,
                    leaf_batch_size=leaf_batch_size,
                    **sf_common
                )
                info["vs_best"] = False
                info["vs_opp"] = True
            else:
                game_samples, res, plies, info = make_game_samples_unified(
                    net, net_black=None, device=device, sims=sims,
                    max_plies=max_plies,
                    resign_threshold=resign_threshold,
                    resign_patience=resign_patience,
                    sf_bootstrap_on_star=True,
                    sf_mate_cp=10000,
                    leaf_batch_size=leaf_batch_size,
                    **sf_common
                )
                info["vs_best"] = False
                info["vs_opp"] = False

            while True:
                try:
                    out_q.put((game_samples, res, plies, info), timeout=15)
                    break
                except queue.Full:
                    if stop_ev.is_set():
                        return
                    time.sleep(0.01)
                except Exception:
                    traceback.print_exc()
                    if stop_ev.is_set():
                        return
                    return

    finally:
        if sf_engine is not None:
            try:
                sf_engine.quit()
            except Exception:
                pass


def resolve_device(requested: str) -> str:
    if not requested.startswith("cuda"):
        return requested

    if not torch.cuda.is_available():
        print("[device] CUDA not available -> using CPU")
        return "cpu"

    try:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        _ = (a @ b).sum()
        torch.cuda.synchronize()
        return requested
    except Exception as e:
        print(f"[device] CUDA probe failed -> using CPU. Reason: {e}")
        return "cpu"
