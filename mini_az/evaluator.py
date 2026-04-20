"""Async evaluator — a one-shot process spawned by the orchestrator at each
`--eval_every` boundary. It loads the latest weights from /dev/shm, runs
games vs Stockfish / random / the lagged opponent, decides whether to promote
to `best.pt`, and writes the eval CSV.

The trainer and self-play workers keep running the whole time — the
evaluator just competes for CPU, sized via `--eval_threads` so impact on
RL throughput stays small. This removes the ~30-60 s pause_ev stall that
blocked workers+trainer every eval_every iters before Etap 3.2.
"""

import os
import time
from datetime import datetime, timezone

import torch

from .config import print
from .encoding import append_eval_csv
from .eval_play import play_vs_stockfish, play_vs_random, play_vs_model
from .network import ChessNet


def _eval_threads(args) -> int:
    n = int(getattr(args, "eval_threads", 0) or 0)
    if n > 0:
        return n
    if args.workers and args.workers > 0:
        return max(4, args.workers // 12)
    return 4


def run_eval_job(args, device: str, it: int, best_path: str,
                  weights_path: str, counters: dict,
                  games_collected: int, rb_len: int) -> None:
    """Entry point for the spawned eval process.

    Loads net from the shared weights file, runs the full eval gate, writes
    CSV and best.pt atomically. Never touches pause_ev — workers and trainer
    keep running.
    """
    torch.set_num_threads(_eval_threads(args))
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    if not os.path.exists(weights_path):
        print(f"[eval it={it}] weights file missing: {weights_path} — skipping eval")
        return

    net = ChessNet().to(device)
    net.use_bf16_inference = bool(getattr(args, "bf16_inference", False))
    try:
        net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"[eval it={it}] could not load weights: {e}")
        return
    net.eval()

    c = counters

    print(f"[eval it={it}] starting {args.eval_games} SF plays (elo={args.sf_eval_elo})")
    sf_score, sf_winrate, sf_elo_diff = play_vs_stockfish(
        net, device, stockfish_path=args.sf_path, sf_skill=args.sf_skill,
        sf_elo=args.sf_eval_elo, sf_movetime_ms=args.sf_movetime_ms,
        games=args.eval_games, sims=args.eval_sims, max_plies=args.sf_eval_max_plies,
    )

    sf_easy_score = 0.0
    sf_easy_winrate = 0.0
    sf_easy_elo_diff = -2400.0
    if args.sf_eval_elo_easy >= 0:
        skill_lvl = args.sf_eval_elo_easy
        print(f"[eval it={it}] SF plays (Skill Level {skill_lvl})")
        sf_easy_score, sf_easy_winrate, sf_easy_elo_diff = play_vs_stockfish(
            net, device, stockfish_path=args.sf_path, sf_skill=args.sf_skill,
            sf_elo=args.sf_eval_elo, sf_movetime_ms=args.sf_movetime_ms,
            games=args.eval_games, sims=args.eval_sims, max_plies=args.sf_eval_max_plies,
            use_skill_level=skill_lvl,
        )

    print(f"[eval it={it}] {args.rand_eval_games} random games")
    rnd_score, rnd_winrate, rnd_elo_diff = play_vs_random(
        net, device, games=args.rand_eval_games,
        sims=args.rand_eval_sims, max_plies=args.rand_max_plies,
    )

    self_score = None
    self_winrate = None
    if os.path.exists(args.opponent_model):
        opp_eval = ChessNet().to(device)
        opp_eval.use_bf16_inference = bool(getattr(args, "bf16_inference", False))
        opp_eval.eval()
        try:
            opp_eval.load_state_dict(torch.load(
                args.opponent_model, map_location=device, weights_only=True
            ))
        except Exception as e:
            print(f"[eval it={it}] opponent model incompatible, skipping self-eval ({e})")
            opp_eval = None
        if opp_eval is not None:
            print(f"[eval it={it}] {args.self_eval_games} vs-opponent games")
            self_score, self_winrate = play_vs_model(
                net, opp_eval, device,
                games=args.self_eval_games,
                sims=args.self_eval_sims,
                max_plies=args.self_eval_max_plies,
            )

    # --- Gating (unchanged logic, kept in sync with _run_eval_gate) ---
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

    SELF_TIE_EPS = 0.02
    SELF_MIN_WIN = 0.02
    metric_self = None
    if self_score is not None:
        sc = self_score
        wr = self_winrate if self_winrate is not None else 0.0
        if (abs(sc - 0.5) >= SELF_TIE_EPS) or (wr >= SELF_MIN_WIN):
            metric_self = sc

    if metric_self is not None:
        gate_kind, metric = "self", metric_self
    elif sf_score > 0.05:
        gate_kind, metric = "sf_score", float(sf_score)
    elif args.sf_eval_elo_easy >= 0 and sf_easy_score > 0.05:
        gate_kind, metric = "sf_easy_score", float(sf_easy_score)
    elif sf_winrate >= 0.02:
        gate_kind, metric = "sf_win", float(sf_winrate)
    elif args.sf_eval_elo_easy >= 0 and sf_easy_winrate >= 0.02:
        gate_kind, metric = "sf_easy_win", float(sf_easy_winrate)
    else:
        gate_kind, metric = "rnd", float(rnd_score)

    prev_kind = _read_best_kind()
    prev_metric = _read_best_metric()
    if (prev_metric is not None) and (not prev_kind):
        prev_kind = gate_kind
        _write_best_kind(prev_kind)

    promote = False
    reason = ""
    if prev_metric is None:
        if gate_kind == "self" and abs(metric - 0.5) < 0.02:
            promote, reason = False, "init_but_self_tie"
        else:
            promote, reason = True, "init"
    elif prev_kind != gate_kind:
        promote, reason = False, f"kind_mismatch(prev={prev_kind}, now={gate_kind})"
    else:
        if metric >= prev_metric + args.gate_margin:
            promote, reason = True, "improved"
        else:
            reason = "not_improved"

    if promote:
        tmp = best_path + f".tmp.{os.getpid()}"
        torch.save(net.state_dict(), tmp)
        os.replace(tmp, best_path)  # atomic — worker readers never see a torn file
        _write_best_metric(metric)
        _write_best_kind(gate_kind)
        print(f"[eval it={it}] PROMOTED to best: metric={metric:.3f} prev={prev_metric} kind={gate_kind}")
    else:
        print(f"[eval it={it}] not promoted (metric={metric:.3f} prev={prev_metric} "
              f"best_kind={prev_kind} now_kind={gate_kind}, reason={reason})")

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
        f"[eval it={it}] done: SF elo={args.sf_eval_elo} score={sf_score:.3f} "
        f"elo_diff={sf_elo_diff:+.0f} | RND score={rnd_score:.3f} "
        f"elo_diff={rnd_elo_diff:+.0f}{easy_str} (logged to {args.eval_csv})"
    )
