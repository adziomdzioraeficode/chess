"""Monte Carlo Tree Search with PUCT selection."""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Hashable

import numpy as np
import chess

from .config import PV, NO_PROGRESS_HALFMOVE, material_score
from .encoding import HISTORY_STEPS
from .network import ChessNet


@dataclass
class EdgeStats:
    P: float
    N: int = 0
    W: float = 0.0


class MCTSNode:
    def __init__(self, board: chess.Board):
        self.board = board
        self.expanded = False
        self.edges: Dict[chess.Move, EdgeStats] = {}
        self.children: Dict[chess.Move, "MCTSNode"] = {}


def mcts_search(
    net: ChessNet,
    root_board: chess.Board,
    device: str,
    sims: int = 200,
    c_puct: float = 2.5,
    fpu_reduction: float = 0.25,
    dirichlet_alpha: float = 0.0,
    dirichlet_eps: float = 0.0,
    history: list = None,
) -> Tuple[Dict[chess.Move, int], float]:
    """MCTS with dynamic c_puct (AlphaZero-style log scaling)."""
    C_BASE = 19652
    C_INIT = c_puct

    root = MCTSNode(root_board.copy())
    pv_cache: dict[Hashable, PV] = {}

    # Build history list for child nodes (track position sequence)
    if history is None:
        history = []

    def _board_key(b: chess.Board) -> Hashable:
        try:
            return b._transposition_key()
        except Exception:
            return b.fen()

    def expand(node: MCTSNode, hist: list) -> float:
        key = _board_key(node.board)
        cached = pv_cache.get(key)
        if cached is None:
            priors, v = net.policy_value_single(node.board, device, history=hist)
            pv_cache[key] = (priors, v)
        else:
            priors, v = cached
        node.edges = {mv: EdgeStats(P=p) for mv, p in priors.items()}
        node.expanded = True
        return v

    def select(node: MCTSNode) -> Tuple[chess.Move, MCTSNode]:
        total_N = sum(e.N for e in node.edges.values()) + 1
        # Dynamic c_puct: increases with total visits (AlphaZero formula)
        c_dyn = math.log((1 + total_N + C_BASE) / C_BASE) + C_INIT

        # Parent's mean Q — used as baseline for FPU
        visited_N = sum(e.N for e in node.edges.values() if e.N > 0)
        if visited_N > 0:
            parent_Q = sum(e.W for e in node.edges.values() if e.N > 0) / visited_N
        else:
            parent_Q = 0.0
        fpu_value = parent_Q - fpu_reduction  # pessimistic prior for unvisited

        best_mv = None
        best_score = -1e9

        for mv, e in node.edges.items():
            Q = fpu_value if e.N == 0 else e.W / e.N
            U = c_dyn * e.P * math.sqrt(total_N) / (1 + e.N)
            score = Q + U
            if score > best_score:
                best_score = score
                best_mv = mv

        assert best_mv is not None
        if best_mv not in node.children:
            b2 = node.board.copy()
            b2.push(best_mv)
            node.children[best_mv] = MCTSNode(b2)
        return best_mv, node.children[best_mv]

    def backup(path: List[Tuple[MCTSNode, chess.Move]], leaf_value: float):
        v = leaf_value
        for node, mv in reversed(path):
            v = -v
            e = node.edges[mv]
            e.N += 1
            e.W += v

    if root.board.is_game_over():
        return {}, 0.0

    root_v: float = expand(root, history)

    if dirichlet_alpha > 0:
        moves = list(root.edges.keys())
        if moves:
            noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
            for i, mv in enumerate(moves):
                e = root.edges[mv]
                e.P = (1 - dirichlet_eps) * e.P + dirichlet_eps * float(noise[i])

    for _ in range(sims):
        node = root
        path: List[Tuple[MCTSNode, chess.Move]] = []
        # Build history for traversal: start from root's history, keep bounded
        trav_hist = list(history[:HISTORY_STEPS]) if history else []
        # An expanded terminal node has no edges, so `len(node.edges) > 0`
        # already guards against re-entering finished positions without a
        # separate is_game_over() call on every traversal step.
        while node.expanded and len(node.edges) > 0:
            mv, child = select(node)
            path.append((node, mv))
            # Add current board to history for the child (bounded)
            trav_hist = [node.board] + trav_hist[:HISTORY_STEPS - 1]
            node = child

        if node.board.is_game_over(claim_draw=False):
            res = node.board.result(claim_draw=False)
            if res == "1-0":
                leaf_v = 1.0 if node.board.turn == chess.WHITE else -1.0
            elif res == "0-1":
                leaf_v = 1.0 if node.board.turn == chess.BLACK else -1.0
            else:
                leaf_v = 0.0
        else:
            leaf_v = expand(node, trav_hist[:HISTORY_STEPS])

        backup(path, leaf_v)

    total = sum(e.N for e in root.edges.values())
    if total > 0:
        root_q = 0.0
        for e in root.edges.values():
            if e.N > 0:
                root_q += (e.N / total) * (e.W / e.N)
    else:
        root_q = root_v

    return {mv: e.N for mv, e in root.edges.items()}, float(root_q)


def pick_move_from_visits(visits: Dict[chess.Move, int], temperature: float) -> chess.Move:
    items = list(visits.items())
    moves = [m for m, _ in items]
    counts = np.array([c for _, c in items], dtype=np.float64)

    if len(moves) == 0:
        raise RuntimeError("No moves to pick from (visits empty).")

    if temperature <= 1e-6:
        return moves[int(np.argmax(counts))]

    counts = counts ** (1.0 / temperature)
    probs = counts / (counts.sum() + 1e-12)
    idx = int(np.random.choice(len(moves), p=probs))
    return moves[idx]


def pick_move_uci(board: chess.Board, visits: Dict[chess.Move, int], *, temperature: float) -> chess.Move:
    legals = list(board.legal_moves)
    if not legals:
        raise RuntimeError("No legal moves in UCI pick_move_uci")

    counts = np.array([visits.get(mv, 0) for mv in legals], dtype=np.float64)
    if counts.sum() == 0:
        counts += 1.0

    ms_now = material_score(board)
    adv = ms_now if board.turn == chess.WHITE else -ms_now
    rep_penalty = 0.35 if adv >= 3 else 0.70

    for i, mv in enumerate(legals):
        b2 = board.copy()
        b2.push(mv)
        if (b2.can_claim_threefold_repetition()
            or b2.can_claim_fifty_moves()
            or b2.halfmove_clock >= NO_PROGRESS_HALFMOVE
            or b2.is_insufficient_material()
            or b2.is_stalemate()):
            counts[i] *= rep_penalty

    counts = counts + 1e-6 * np.random.random(size=len(counts))

    if temperature <= 1e-6:
        return legals[int(np.argmax(counts))]

    counts = counts ** (1.0 / temperature)
    probs = counts / (counts.sum() + 1e-12)
    idx = int(np.random.choice(len(legals), p=probs))
    return legals[idx]
