#!/usr/bin/env python3
"""Elo benchmark: play matches between lc0/Maia, our model, and SF."""

import argparse
import math
import os
import sys
import time

import chess
import chess.engine
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mini_az.network import ChessNet
from mini_az.encoding import HISTORY_STEPS
from mini_az.mcts import mcts_search, pick_move_uci


def elo_diff(score: float) -> float:
    if score <= 0.001:
        return -999.0
    if score >= 0.999:
        return 999.0
    return -400.0 * math.log10(1.0 / score - 1.0)


class MiniAzPlayer:
    """Wrapper for our network + MCTS."""
    def __init__(self, model_path: str, sims: int = 200, device: str = "cpu"):
        self.net = ChessNet()
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.net.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and all(k.startswith(("conv", "res", "pol", "val", "ml")) for k in list(ckpt.keys())[:5]):
            self.net.load_state_dict(ckpt)
        else:
            self.net.load_state_dict(ckpt)
        self.net.eval()
        self.device = device
        self.sims = sims

    def pick_move(self, board: chess.Board, history: list) -> chess.Move:
        visits, _ = mcts_search(
            self.net, board, self.device, sims=self.sims,
            policy_temp=1.0, dirichlet_alpha=0.0, dirichlet_eps=0.0,
            history=history[:HISTORY_STEPS],
        )
        if not visits:
            return list(board.legal_moves)[0]
        return pick_move_uci(board, visits, temperature=1e-6)


class UciPlayer:
    """Wrapper for UCI engines with a fixed Limit."""
    def __init__(self, engine: chess.engine.SimpleEngine, limit: chess.engine.Limit):
        self.engine = engine
        self.limit = limit

    def quit(self):
        self.engine.quit()


def play_match(
    player_a,
    player_b,
    name_a: str,
    name_b: str,
    games: int = 10,
    time_limit_ms: int = 500,
    max_plies: int = 300,
) -> dict:
    W = D = L = 0

    def get_move(player, board, history):
        if isinstance(player, UciPlayer):
            r = player.engine.play(board, player.limit)
            return r.move
        elif isinstance(player, MiniAzPlayer):
            return player.pick_move(board, history)
        else:
            raise ValueError(f"Unknown player type: {type(player)}")

    for g in range(games):
        if g % 2 == 0:
            white, black = player_a, player_b
            wname, bname = name_a, name_b
        else:
            white, black = player_b, player_a
            wname, bname = name_b, name_a

        board = chess.Board()
        history: list[chess.Board] = []
        ply = 0

        while not board.is_game_over(claim_draw=True) and ply < max_plies:
            player = white if board.turn == chess.WHITE else black
            try:
                mv = get_move(player, board, history)
                if mv is None:
                    break
                history = [board.copy()] + history[:HISTORY_STEPS - 1]
                board.push(mv)
            except Exception as e:
                print(f"  Error at ply {ply}: {e}")
                break
            ply += 1

        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)
        else:
            res = "1/2-1/2"

        if g % 2 == 0:
            if res == "1-0":   W += 1
            elif res == "0-1": L += 1
            else:              D += 1
        else:
            if res == "1-0":   L += 1
            elif res == "0-1": W += 1
            else:              D += 1

        score_so_far = (W + 0.5 * D) / (g + 1)
        print(f"  Game {g+1}/{games}: {wname}(W) vs {bname}(B) = {res} ply={ply}  "
              f"[{name_a}: +{W}-{L}={D} score={score_so_far:.3f}]")

    total = W + D + L
    score = (W + 0.5 * D) / max(1, total)
    return {"w": W, "d": D, "l": L, "score": score, "games": total}


def main():
    parser = argparse.ArgumentParser(description="Elo benchmark")
    parser.add_argument("--games", type=int, default=6, help="Games per match")
    parser.add_argument("--time-ms", type=int, default=200, help="Time per move for UCI engines (ms)")
    parser.add_argument("--maia-path", default="lc0_nets/maia-1100.pb.gz")
    parser.add_argument("--maia-nodes", type=int, default=1)
    parser.add_argument("--bg1-path", default="lc0_nets/bad-gyal-1.pb.gz")
    parser.add_argument("--bg1-nodes", type=int, default=1)
    parser.add_argument("--model-path", default="models/best.pt")
    parser.add_argument("--model-sims", type=int, default=200, help="MCTS sims for our model")
    parser.add_argument("--sf-path", default="/usr/games/stockfish")
    parser.add_argument("--sf-elo", type=int, default=1320, help="SF limited Elo")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--skip-sf", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    print("=" * 60)
    print("ELO BENCHMARK")
    print("=" * 60)

    print(f"\nLoading mini_az: {args.model_path} (sims={args.model_sims})")
    mini = MiniAzPlayer(args.model_path, sims=args.model_sims)

    print(f"Loading Maia-1100: {args.maia_path} (nodes={args.maia_nodes})")
    maia_engine = chess.engine.SimpleEngine.popen_uci(
        ["lc0", f"--weights={args.maia_path}", "--threads=1", "--verbose-move-stats=false"]
    )
    maia = UciPlayer(maia_engine, chess.engine.Limit(nodes=args.maia_nodes))

    bg1 = None
    if args.bg1_path and os.path.isfile(args.bg1_path):
        print(f"Loading Bad Gyal 1: {args.bg1_path} (nodes={args.bg1_nodes})")
        bg1_engine = chess.engine.SimpleEngine.popen_uci(
            ["lc0", f"--weights={args.bg1_path}", "--threads=1", "--verbose-move-stats=false"]
        )
        bg1 = UciPlayer(bg1_engine, chess.engine.Limit(nodes=args.bg1_nodes))

    sf_label = f"SF-{args.sf_elo}"
    if not args.skip_sf:
        print(f"Loading Stockfish: {args.sf_path} (elo={args.sf_elo})")
        sf_engine = chess.engine.SimpleEngine.popen_uci(args.sf_path)
        sf_engine.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": args.sf_elo})
        sf = UciPlayer(sf_engine, chess.engine.Limit(time=args.time_ms / 1000.0))

    results = {}

    if not args.skip_sf:
        print(f"\n{'='*60}")
        print(f"MATCH 1: Maia-1100 vs {sf_label} ({args.games} games)")
        print(f"{'='*60}")
        t0 = time.time()
        r = play_match(maia, sf, "Maia-1100", sf_label,
                        games=args.games, time_limit_ms=args.time_ms, max_plies=args.max_plies)
        results[("Maia-1100", sf_label)] = r
        print(f"\nMaia-1100 vs {sf_label}: +{r['w']}-{r['l']}={r['d']} "
              f"score={r['score']:.3f} Elo={elo_diff(r['score']):+.0f} ({time.time()-t0:.0f}s)")

    if not args.skip_sf:
        print(f"\n{'='*60}")
        print(f"MATCH 2: mini_az vs {sf_label} ({args.games} games)")
        print(f"{'='*60}")
        t0 = time.time()
        r = play_match(mini, sf, "mini_az", sf_label,
                        games=args.games, time_limit_ms=args.time_ms, max_plies=args.max_plies)
        results[("mini_az", sf_label)] = r
        print(f"\nmini_az vs {sf_label}: +{r['w']}-{r['l']}={r['d']} "
              f"score={r['score']:.3f} Elo={elo_diff(r['score']):+.0f} ({time.time()-t0:.0f}s)")

    print(f"\n{'='*60}")
    print(f"MATCH 3: mini_az vs Maia-1100 ({args.games} games)")
    print(f"{'='*60}")
    t0 = time.time()
    r = play_match(mini, maia, "mini_az", "Maia-1100",
                    games=args.games, time_limit_ms=args.time_ms, max_plies=args.max_plies)
    results[("mini_az", "Maia-1100")] = r
    print(f"\nmini_az vs Maia-1100: +{r['w']}-{r['l']}={r['d']} "
          f"score={r['score']:.3f} Elo={elo_diff(r['score']):+.0f} ({time.time()-t0:.0f}s)")

    if bg1 is not None:
        print(f"\n{'='*60}")
        print(f"MATCH 4: mini_az vs Bad Gyal 1 ({args.games} games)")
        print(f"{'='*60}")
        t0 = time.time()
        r = play_match(mini, bg1, "mini_az", "BG1",
                        games=args.games, time_limit_ms=args.time_ms, max_plies=args.max_plies)
        results[("mini_az", "BG1")] = r
        print(f"\nmini_az vs BG1: +{r['w']}-{r['l']}={r['d']} "
              f"score={r['score']:.3f} Elo={elo_diff(r['score']):+.0f} ({time.time()-t0:.0f}s)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY & ELO ESTIMATES")
    print(f"{'='*60}")

    anchor_elo = args.sf_elo if not args.skip_sf else 1100
    anchor_name = sf_label if not args.skip_sf else "Maia-1100"
    elo_est = {anchor_name: anchor_elo}

    for _ in range(2):
        for (a, b), r in results.items():
            diff = elo_diff(r['score'])
            if a in elo_est and b not in elo_est:
                elo_est[b] = elo_est[a] - diff
            elif b in elo_est and a not in elo_est:
                elo_est[a] = elo_est[b] + diff

    print(f"\nAnchor: {anchor_name} = {anchor_elo} Elo\n")
    print("Estimated ratings:")
    for name, elo in sorted(elo_est.items(), key=lambda x: -x[1]):
        marker = " <-- our model" if name == "mini_az" else ""
        print(f"  {name:20s}  {elo:7.0f}{marker}")

    print("\nMatch details:")
    for (a, b), r in results.items():
        print(f"  {a} vs {b}: +{r['w']}-{r['l']}={r['d']} "
              f"score={r['score']:.3f} Elo diff={elo_diff(r['score']):+.0f}")

    maia.quit()
    if bg1 is not None:
        bg1.quit()
    if not args.skip_sf:
        sf.quit()

    print("\nDone!")


if __name__ == "__main__":
    main()
