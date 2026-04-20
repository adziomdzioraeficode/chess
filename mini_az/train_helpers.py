"""Shared iteration-level helpers used by both the orchestrator (main.py)
and the async trainer process (trainer.py). Pulled out of main.py so the
trainer can import them without dragging main's argparse/eval code along.
"""

import os
import shutil
from datetime import datetime, timezone

import numpy as np
import torch

from .config import print
from .encoding import append_eval_csv
from .checkpoint import save_checkpoint


def init_counters() -> dict:
    return {
        "pi_ent_sum": 0.0, "v_sum": 0.0,
        "vs_best_games": 0, "vs_opp_games": 0,
        "vmin": 1e9, "vmax": -1e9,
        "threefold_sum": 0,
        "forced_end_sum": 0, "forced_resign_sum": 0, "forced_draw_sum": 0,
        "rep_moves_sum": 0, "rep_plies_sum": 0, "rep_plies_actual_sum": 0,
        "sf_fail_sum": 0,
        "openbook_sum": 0, "ended_star_sum": 0,
        "sf_boot_sum": 0, "sf_cp_sum": 0,
        "sf_cp_used_sum": 0, "sf_cp_used_n": 0,
        "sf_cp_min": 10**9, "sf_cp_max": -10**9,
        "sf_cp_used_min": 10**9, "sf_cp_used_max": -10**9,
        "sf_cp_abs_sum": 0, "sf_cp_used_abs_sum": 0,
        "sf_clipped_n": 0, "sf_cp_raw_abs_gt_cap_sum": 0,
        "sf_z_sum": 0.0, "sf_boot_n": 0,
        "sf_boot_star_sum": 0, "sf_boot_forced_draw_sum": 0, "sf_boot_other_sum": 0,
        "forced_res_str_counts": {},
    }


def accumulate_info(info: dict, counters: dict, args) -> None:
    """Update iteration counters from a single game's info dict."""
    c = counters
    c["pi_ent_sum"] += info.get("avg_pi_ent", 0.0)
    c["v_sum"] += info.get("avg_v", 0.0)
    c["vmin"] = min(c["vmin"], info.get("min_v", c["vmin"]))
    c["vmax"] = max(c["vmax"], info.get("max_v", c["vmax"]))
    c["threefold_sum"] += int(info.get("threefold_penalized", 0))
    c["openbook_sum"] += int(info.get("opening_book_plies", 0))
    c["ended_star_sum"] += 1 if info.get("ended_star", False) else 0
    c["rep_moves_sum"] += info.get("rep_moves_penalized", 0)
    c["rep_plies_sum"] += info.get("rep_plies_with_rep", 0)
    c["rep_plies_actual_sum"] += info.get("rep_plies_actual", 0)
    c["sf_fail_sum"] += info.get("sf_fail", 0)

    if info.get("forced_end", False):
        c["forced_end_sum"] += 1
        k = info.get("forced_kind", "")
        if k == "resign":
            c["forced_resign_sum"] += 1
        elif k == "draw_claim":
            c["forced_draw_sum"] += 1

    if info.get("sf_boot_used", False):
        c["sf_z_sum"] += float(info.get("sf_z_white", 0.0))
        c["sf_boot_sum"] += 1
        c["sf_boot_n"] += 1
        c["sf_cp_used_n"] += 1

        raw = int(info.get("sf_cp_white", 0))
        used = int(info.get("sf_cp_used", 0))
        cap_used = int(args.sf_cp_cap)

        c["sf_cp_sum"] += raw
        c["sf_cp_used_sum"] += used
        c["sf_cp_abs_sum"] += abs(raw)
        c["sf_cp_used_abs_sum"] += abs(used)

        c["sf_cp_min"] = min(c["sf_cp_min"], raw)
        c["sf_cp_max"] = max(c["sf_cp_max"], raw)
        c["sf_cp_used_min"] = min(c["sf_cp_used_min"], used)
        c["sf_cp_used_max"] = max(c["sf_cp_used_max"], used)

        if cap_used > 0 and abs(raw) > cap_used:
            c["sf_clipped_n"] += 1
            c["sf_cp_raw_abs_gt_cap_sum"] += 1

        src = info.get("sf_boot_src", "")
        if src == "star":
            c["sf_boot_star_sum"] += 1
        elif src == "forced_draw":
            c["sf_boot_forced_draw_sum"] += 1
        else:
            c["sf_boot_other_sum"] += 1

    frs = info.get("forced_res_str", "")
    if frs:
        c["forced_res_str_counts"][frs] = c["forced_res_str_counts"].get(frs, 0) + 1

    c["vs_best_games"] += 1 if info.get("vs_best") else 0
    c["vs_opp_games"] += 1 if info.get("vs_opp") else 0


def print_selfplay_info(it, c, gc):
    avg_openbook = c["openbook_sum"] / gc
    ended_star_rate = c["ended_star_sum"] / gc
    sf_boot_rate = c["sf_boot_sum"] / gc
    sf_boot_n = c["sf_boot_n"]
    sf_cp_used_n = c["sf_cp_used_n"]
    sf_cp_mean = (c["sf_cp_sum"] / sf_boot_n) if sf_boot_n else 0.0
    sf_z_mean = (c["sf_z_sum"] / sf_boot_n) if sf_boot_n else 0.0
    sf_cp_used_mean = (c["sf_cp_used_sum"] / sf_cp_used_n) if sf_cp_used_n else 0.0
    sf_cp_abs_mean = (c["sf_cp_abs_sum"] / sf_boot_n) if sf_boot_n else 0.0
    sf_cp_used_abs_mean = (c["sf_cp_used_abs_sum"] / sf_boot_n) if sf_boot_n else 0.0
    sf_clipped_rate = (c["sf_clipped_n"] / sf_boot_n) if sf_boot_n else 0.0
    sf_cp_raw_gt_cap_rate = (c["sf_cp_raw_abs_gt_cap_sum"] / sf_cp_used_n) if sf_cp_used_n else 0.0

    frs_top = sorted(c["forced_res_str_counts"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    frs_str = ",".join([f"{k}:{v}" for k, v in frs_top]) if frs_top else "-"
    print(
        f"[iter {it}] selfplay info: "
        f"avg_pi_ent={c['pi_ent_sum']/gc:.2f} "
        f"avg_v={c['v_sum']/gc:+.3f} "
        f"v_range=[{c['vmin']:+.2f},{c['vmax']:+.2f}] "
        f"threefold_pen/game={c['threefold_sum']/gc:.2f} "
        f"rep_moves_pen/game={c['rep_moves_sum']/gc:.2f} "
        f"rep_plies_actual_per_game={c['rep_plies_actual_sum']/gc:.2f} "
        f"sf_fail_sum={c['sf_fail_sum']} "
        f"openbook_plies_avg={avg_openbook:.2f} "
        f"ended_star_rate={ended_star_rate:.1%} "
        f"sf_boot_rate={sf_boot_rate:.1%} "
        f"sf_cp_mean={sf_cp_mean:+.1f} "
        f"sf_cp_abs_mean={sf_cp_abs_mean:.1f} "
        f"sf_cp_used_mean={sf_cp_used_mean:+.1f} "
        f"sf_clipped_rate={sf_clipped_rate:.1%} "
        f"sf_z_mean={sf_z_mean:+.3f} "
        f"forced_end_rate={c['forced_end_sum']/gc:.1%} "
        f"forced_resign_rate={c['forced_resign_sum']/gc:.1%} "
        f"forced_draw_rate={c['forced_draw_sum']/gc:.1%} "
        f"forced_res_top={frs_str}"
    )


def print_z_stats(it, rb):
    idx = np.random.randint(0, len(rb), size=1000)
    zs = np.array([rb.data[i].z for i in idx], dtype=np.float32)
    mean = float(zs.mean())
    std = float(zs.std())
    zmin = float(zs.min())
    zmax = float(zs.max())
    p_zero = float((np.abs(zs) < 1e-6).mean())
    p_pos = float((zs > 1e-6).mean())
    p_neg = float((zs < -1e-6).mean())
    print(
        f"[iter {it}] z stats (n=1000): "
        f"mean={mean:+.3f} std={std:.3f} min={zmin:+.2f} max={zmax:+.2f} | "
        f"%zero={p_zero*100:.1f}% %pos={p_pos*100:.1f}% %neg={p_neg*100:.1f}%"
    )


def log_iteration_csv(args, it, c, games_collected, res_counts, avg_plies,
                        new_samples, rb_len, last_m, t_sp, t_tr, t_iter, cur_lr):
    """Write one row to train_log.csv with all iteration metrics."""
    gc = max(1, games_collected)
    sf_boot_n = max(1, c["sf_boot_n"]) if c["sf_boot_n"] > 0 else 1
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iter": it,
        "games": games_collected,
        "new_samples": new_samples,
        "buffer_len": rb_len,
        "avg_plies": round(avg_plies, 1),
        "w_1_0": res_counts.get("1-0", 0),
        "w_0_1": res_counts.get("0-1", 0),
        "draws": res_counts.get("1/2-1/2", 0),
        "stars": res_counts.get("*", 0),
        "avg_pi_ent": round(c["pi_ent_sum"] / gc, 3),
        "avg_v": round(c["v_sum"] / gc, 4),
        "v_min": round(c["vmin"], 3) if c["vmin"] < 1e8 else "",
        "v_max": round(c["vmax"], 3) if c["vmax"] > -1e8 else "",
        "forced_end_rate": round(c["forced_end_sum"] / gc, 4),
        "forced_resign_rate": round(c["forced_resign_sum"] / gc, 4),
        "forced_draw_rate": round(c["forced_draw_sum"] / gc, 4),
        "sf_boot_rate": round(c["sf_boot_sum"] / gc, 4),
        "sf_z_mean": round(c["sf_z_sum"] / sf_boot_n, 4) if c["sf_boot_n"] > 0 else "",
        "sf_cp_abs_mean": round(c["sf_cp_abs_sum"] / sf_boot_n, 1) if c["sf_boot_n"] > 0 else "",
        "sf_fail_sum": c["sf_fail_sum"],
        "vs_best_rate": round(c["vs_best_games"] / gc, 3),
        "vs_opp_rate": round(c["vs_opp_games"] / gc, 3),
        "threefold_per_game": round(c["threefold_sum"] / gc, 2),
        "rep_plies_per_game": round(c["rep_plies_actual_sum"] / gc, 2),
        "loss": round(last_m["loss"], 5) if last_m else "",
        "pol_loss": round(last_m["pol"], 5) if last_m else "",
        "val_loss": round(last_m["val"], 5) if last_m else "",
        "ml_loss": round(last_m["ml"], 5) if last_m else "",
        "grad_norm": round(last_m["grad_norm"], 3) if last_m else "",
        "pred_ent": round(last_m["pred_ent"], 3) if last_m else "",
        "tgt_ent": round(last_m["tgt_ent"], 3) if last_m else "",
        "v_mean": round(last_m["v_mean"], 4) if last_m else "",
        "z_mean": round(last_m["z_mean"], 4) if last_m else "",
        "vz_corr": round(last_m["vz_corr"], 4) if last_m else "",
        "ml_mean": round(last_m["ml_mean"], 2) if last_m else "",
        "ml_tgt_mean": round(last_m["ml_tgt_mean"], 2) if last_m else "",
        "lr": f"{cur_lr:.2e}",
        "selfplay_time_s": round(t_sp, 1),
        "train_time_s": round(t_tr, 1),
        "iter_time_s": round(t_iter, 1),
        "samples_per_s": round(new_samples / max(1e-9, t_sp), 1),
        "games_per_s": round(games_collected / max(1e-9, t_sp), 2),
    }
    train_csv = os.path.join(os.path.dirname(args.eval_csv) or ".", "train_log.csv")
    append_eval_csv(train_csv, row)


def save_iter(args, net, opt, it, rb, best_path):
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"mini_az_iter_{it:05d}.pt")
    # Atomic writes (tmp + rename) so readers never see a torn file.
    tmp_snap = path + f".tmp.{os.getpid()}"
    torch.save(net.state_dict(), tmp_snap)
    os.replace(tmp_snap, path)

    tmp_main = args.model + f".tmp.{os.getpid()}"
    torch.save(net.state_dict(), tmp_main)
    os.replace(tmp_main, args.model)

    if args.ckpt:
        save_checkpoint(args.ckpt, net, opt, it)
    if args.replay_path:
        rb.dump(args.replay_path)
    print(f"[iter {it}] saved weights={args.model} ckpt={args.ckpt} replay={args.replay_path}")

    # Update opponent snapshot from a lagged iteration's weights.
    opp_path = args.opponent_model
    os.makedirs(os.path.dirname(opp_path) or ".", exist_ok=True)
    lag_it = it - int(args.opp_lag)
    if lag_it > 0:
        SEARCH_BACK_MAX = max(200, int(args.opp_lag) * 4)
        cand = None
        for j in range(lag_it, max(0, lag_it - SEARCH_BACK_MAX), -1):
            p = os.path.join(args.save_dir, f"mini_az_iter_{j:05d}.pt")
            if os.path.exists(p):
                cand = p
                break
        if cand is not None:
            same = False
            try:
                if os.path.exists(opp_path):
                    st_o = os.stat(opp_path)
                    st_c = os.stat(cand)
                    same = (st_o.st_size == st_c.st_size) and (int(st_o.st_mtime) == int(st_c.st_mtime))
            except Exception:
                same = False
            if not same:
                tmp = opp_path + f".tmp.{os.getpid()}"
                shutil.copy2(cand, tmp)
                os.replace(tmp, opp_path)
                print(f"[iter {it}] updated opponent snapshot: {opp_path} <- {cand}")
        else:
            print(f"[iter {it}] WARNING: no snapshot found for opponent (wanted <= {lag_it})")
