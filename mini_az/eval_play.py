"""Evaluation: play vs Stockfish, random, other model; UCI loop."""

import random
from typing import Dict, Tuple, Optional

import chess
import chess.engine

from .config import print, apply_random_opening, material_score
from .network import ChessNet
from .encoding import HISTORY_STEPS
from .mcts import mcts_search, pick_move_from_visits, pick_move_uci
from .stockfish import open_stockfish_engine, elo_diff_from_score


def play_vs_stockfish(
    net: ChessNet,
    device: str,
    stockfish_path: str = "stockfish",
    sf_skill: int = 0,
    sf_movetime_ms: int = 50,
    games: int = 20,
    sims: int = 200,
    sf_elo: int = 1320,
    max_plies: int = 300,
    use_skill_level: int | None = None,
) -> Tuple[float, float, float]:
    if use_skill_level is not None:
        engine = open_stockfish_engine(
            stockfish_path,
            threads=1,
            hash_mb=16,
            elo=None,
            skill=use_skill_level
        )
    else:
        engine = open_stockfish_engine(
            stockfish_path,
            threads=1,
            hash_mb=16,
            elo=sf_elo,
            skill=None
        )

    W = D = L = 0
    try:
        for g in range(games):
            print(f"[eval] game {g+1}/{games}")
            board = chess.Board()
            apply_random_opening(board, max_book_plies=8)
            our_color = chess.WHITE if (g % 2 == 0) else chess.BLACK
            forced_result: Optional[str] = None
            board_history: list = []

            while not board.is_game_over(claim_draw=False) and board.ply() < max_plies:
                if board.ply() > 0 and (board.ply() % 40 == 0):
                    print(f"[eval] game {g+1}/{games} ply={board.ply()}")
                if board.turn == our_color:
                    visits, v_now = mcts_search(
                        net, board, device, sims=sims,
                        dirichlet_alpha=0.0,
                        dirichlet_eps=0.0,
                        history=board_history[:HISTORY_STEPS])
                    if not visits:
                        break
                    mv = pick_move_uci(board, visits, temperature=1e-6)
                    board_history = [board.copy()] + board_history[:HISTORY_STEPS - 1]
                    board.push(mv)
                else:
                    r = engine.play(board, chess.engine.Limit(time=sf_movetime_ms / 1000.0))

                    if r.move is None:
                        if getattr(r, "resigned", False):
                            forced_result = "1-0" if our_color == chess.WHITE else "0-1"
                        else:
                            forced_result = "1/2-1/2"
                        break

                    board_history = [board.copy()] + board_history[:HISTORY_STEPS - 1]
                    board.push(r.move)

            if forced_result is not None:
                res = forced_result
            elif board.is_game_over(claim_draw=True):
                res = board.result(claim_draw=True)
            else:
                ms = material_score(board)
                if ms >= 3:      res = "1-0"
                elif ms <= -3:   res = "0-1"
                else:            res = "1/2-1/2"

            print(
                f"[eval] game {g+1}/{games} END our={'W' if our_color==chess.WHITE else 'B'} "
                f"res={res} forced={forced_result} plies={board.ply()} halfmove={board.halfmove_clock}"
            )

            game_over_no_claim = board.is_game_over(claim_draw=False)
            game_over_claim = board.is_game_over(claim_draw=True)

            print(
                f"[eval] finished: over_no_claim={game_over_no_claim} over_claim={game_over_claim} "
                f"fen={board.fen()}"
            )

            if game_over_claim:
                print(f"[eval] official result(claim)= {board.result(claim_draw=True)}")
            else:
                ms_now = material_score(board)
                print(f"[eval] not over -> adjudicate by material ms={ms_now}")

            if res == "1-0":
                if our_color == chess.WHITE: W += 1
                else: L += 1
            elif res == "0-1":
                if our_color == chess.BLACK: W += 1
                else: L += 1
            else:
                D += 1
            print(f"[eval] W={W} D={D} L={L} score={(W+0.5*D)/(g+1):.3f}")

        score = (W + 0.5 * D) / max(1, games)
        winrate = W / max(1, games)
        ed = elo_diff_from_score(score)
    finally:
        engine.quit()
    return score, winrate, ed


def play_vs_random(
    net: ChessNet,
    device: str,
    games: int = 50,
    sims: int = 200,
    max_plies: int = 250,
    min_games: int = 20,
    stop_margin: float = 0.03
) -> Tuple[float, float, float]:
    W = D = L = 0

    for g in range(games):
        board = chess.Board()
        our_color = chess.WHITE if (g % 2 == 0) else chess.BLACK
        board_history: list = []

        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < max_plies:
            if board.turn == our_color:
                visits, v_now = mcts_search(
                    net, board, device, sims=sims,
                    dirichlet_alpha=0.0,
                    dirichlet_eps=0.0,
                    history=board_history[:HISTORY_STEPS]
                )
                if not visits:
                    break
                mv = pick_move_uci(board, visits, temperature=1e-6)
                board_history = [board.copy()] + board_history[:HISTORY_STEPS - 1]
                board.push(mv)
            else:
                legals = list(board.legal_moves)
                if not legals:
                    break
                board_history = [board.copy()] + board_history[:HISTORY_STEPS - 1]
                board.push(random.choice(legals))

            plies += 1

        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)
        else:
            ms = material_score(board)
            if ms >= 3:      res = "1-0"
            elif ms <= -3:   res = "0-1"
            else:            res = "1/2-1/2"

        if res == "1-0":
            if our_color == chess.WHITE: W += 1
            else: L += 1
        elif res == "0-1":
            if our_color == chess.BLACK: W += 1
            else: L += 1
        else:
            D += 1

        played = W + D + L
        if played >= min_games:
            score = (W + 0.5 * D) / played
            if score > 0.5 + stop_margin or score < 0.5 - stop_margin:
                break

    played = W + D + L
    score = (W + 0.5 * D) / max(1, played)
    winrate = W / max(1, played)
    elo_diff = elo_diff_from_score(score)
    return score, winrate, elo_diff


def play_vs_model(
    net_a: ChessNet,
    net_b: ChessNet,
    device: str,
    games: int = 50,
    sims: int = 100,
    max_plies: int = 300,
) -> Tuple[float, float]:
    Wa = Da = La = 0
    for g in range(games):
        board = chess.Board()
        apply_random_opening(board, max_book_plies=8)
        a_color = chess.WHITE if (g % 2 == 0) else chess.BLACK
        board_history: list = []

        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            net_to_move = net_a if (board.turn == a_color) else net_b
            visits, _ = mcts_search(
                net_to_move, board, device, sims=sims,
                dirichlet_alpha=0.0, dirichlet_eps=0.0,
                history=board_history[:HISTORY_STEPS]
            )
            if not visits:
                break
            mv = pick_move_uci(board, visits, temperature=1e-6)
            board_history = [board.copy()] + board_history[:HISTORY_STEPS - 1]
            board.push(mv)

        res = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        if res == "1-0":
            if a_color == chess.WHITE: Wa += 1
            else: La += 1
        elif res == "0-1":
            if a_color == chess.BLACK: Wa += 1
            else: La += 1
        else:
            Da += 1

    score = (Wa + 0.5 * Da) / max(1, games)
    winrate = Wa / max(1, games)
    return score, winrate


def uci_loop(net: ChessNet, device: str, sims: int):
    board = chess.Board()
    board_history: list = []

    def send(msg: str):
        print(msg)

    send("id name mini_az_chess")
    send("id author you")
    send("uciok")

    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            continue

        if line == "isready":
            send("readyok")
        elif line == "ucinewgame":
            board = chess.Board()
            board_history = []
        elif line.startswith("position"):
            tokens = line.split()
            board_history = []
            if "startpos" in tokens:
                board = chess.Board()
                if "moves" in tokens:
                    mi = tokens.index("moves")
                    for u in tokens[mi + 1:]:
                        board_history = [board.copy()] + board_history[:HISTORY_STEPS - 1]
                        board.push_uci(u)
            else:
                if "fen" in tokens:
                    fi = tokens.index("fen")
                    if "moves" in tokens:
                        mi = tokens.index("moves")
                        fen = " ".join(tokens[fi + 1:mi])
                        moves = tokens[mi + 1:]
                    else:
                        fen = " ".join(tokens[fi + 1:])
                        moves = []
                    board = chess.Board(fen)
                    for u in moves:
                        board_history = [board.copy()] + board_history[:HISTORY_STEPS - 1]
                        board.push_uci(u)

        elif line.startswith("go"):
            if board.is_game_over():
                send("bestmove 0000")
                continue
            visits, v_now = mcts_search(
                net, board, device, sims=sims,
                dirichlet_alpha=0.0,
                dirichlet_eps=0.0,
                history=board_history[:HISTORY_STEPS])
            if not visits:
                send("bestmove 0000")
                continue
            temp_moves = 12
            temperature = 1.0 if board.ply() < temp_moves else 1e-6
            mv = pick_move_uci(board, visits, temperature=temperature)
            send(f"bestmove {mv.uci()}")

        elif line == "quit":
            break
