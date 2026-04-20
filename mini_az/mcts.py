"""Monte Carlo Tree Search with PUCT selection and virtual-loss leaf batching.

With `leaf_batch_size > 1`, traversals apply virtual loss to in-flight edges so
parallel simulations explore distinct paths; the collected K leaves are evaluated
in a single batched forward pass, then real values are committed (virtual loss
reverted). This amortizes per-forward Python/kernel overhead on CPU and is the
main `mcts_search` throughput knob on EPYC 9004.
"""

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
    # In-flight "virtual loss" visits held by sims that started traversing
    # this edge but haven't returned from the batched network eval yet.
    # PUCT treats them as pending losses for the selecting player.
    N_virt: int = 0


class MCTSNode:
    def __init__(self, board: chess.Board):
        self.board = board
        self.expanded = False
        self.edges: Dict[chess.Move, EdgeStats] = {}
        self.children: Dict[chess.Move, "MCTSNode"] = {}


def _terminal_leaf_value(board: chess.Board) -> float:
    """Value from side-to-move perspective for a terminal board."""
    res = board.result(claim_draw=False)
    if res == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    if res == "0-1":
        return 1.0 if board.turn == chess.BLACK else -1.0
    return 0.0


def mcts_search(
    net: ChessNet,
    root_board: chess.Board,
    device: str,
    sims: int = 200,
    c_puct: float = 2.5,
    fpu_reduction: float = 0.25,
    fpu_root: float = -1.0,
    policy_temp: float = 1.0,
    dirichlet_alpha: float = 0.0,
    dirichlet_eps: float = 0.0,
    history: list = None,
    leaf_batch_size: int = 1,
) -> Tuple[Dict[chess.Move, int], float]:
    """MCTS with dynamic c_puct (AlphaZero-style log scaling).

    If *leaf_batch_size* > 1, each outer loop gathers up to K unexpanded leaves
    using virtual loss, evaluates them in one batched forward pass, then commits
    their real values. Terminal leaves are short-circuited (no eval needed).
    """
    C_BASE = 19652
    C_INIT = c_puct

    root = MCTSNode(root_board.copy())
    PV_CACHE_MAX = 20_000
    pv_cache: dict[Hashable, PV] = {}

    if history is None:
        history = []

    def _board_key(b: chess.Board) -> Hashable:
        try:
            return b._transposition_key()
        except Exception:
            return b.fen()

    def _cache_put(key, priors_v):
        if len(pv_cache) >= PV_CACHE_MAX:
            pv_cache.pop(next(iter(pv_cache)))
        pv_cache[key] = priors_v

    def _expand_with(node: MCTSNode, priors: dict, v: float) -> float:
        node.edges = {mv: EdgeStats(P=p) for mv, p in priors.items()}
        node.expanded = True
        return v

    def select(node: MCTSNode, is_root: bool = False) -> Tuple[chess.Move, MCTSNode]:
        # Effective visit count includes in-flight virtual visits — they raise
        # the denominator in PUCT and are treated as pending losses in Q.
        total_N = sum(e.N + e.N_virt for e in node.edges.values()) + 1
        c_dyn = math.log((1 + total_N + C_BASE) / C_BASE) + C_INIT

        if is_root:
            fpu_value = fpu_root
        else:
            # FPU uses real visits only (virtual losses ignored here —
            # otherwise it would punish still-settling children twice).
            visited_N = sum(e.N for e in node.edges.values() if e.N > 0)
            if visited_N > 0:
                parent_Q = sum(e.W for e in node.edges.values() if e.N > 0) / visited_N
            else:
                parent_Q = 0.0
            fpu_value = parent_Q - fpu_reduction

        best_mv = None
        best_score = -1e9
        for mv, e in node.edges.items():
            eff_N = e.N + e.N_virt
            if eff_N == 0:
                Q = fpu_value
            else:
                # Pending virtuals count as losses (-1) from selecting player's view.
                Q = (e.W - e.N_virt) / eff_N
            U = c_dyn * e.P * math.sqrt(total_N) / (1 + eff_N)
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

    def descend_to_leaf() -> Tuple[List[Tuple[MCTSNode, chess.Move]], MCTSNode, list]:
        """Walk from root to an unexpanded/terminal leaf, applying virtual loss."""
        path: List[Tuple[MCTSNode, chess.Move]] = []
        trav_hist = list(history[:HISTORY_STEPS])
        node = root
        while node.expanded and len(node.edges) > 0:
            mv, child = select(node, is_root=(node is root))
            e = node.edges[mv]
            e.N_virt += 1  # virtual loss: claim the edge for this in-flight sim
            path.append((node, mv))
            trav_hist = [node.board] + trav_hist[:HISTORY_STEPS - 1]
            node = child
        return path, node, trav_hist[:HISTORY_STEPS]

    def commit(path: List[Tuple[MCTSNode, chess.Move]], leaf_value: float):
        """Revert virtual loss and apply the real leaf_value with sign flipping."""
        v = leaf_value
        for node, mv in reversed(path):
            v = -v
            e = node.edges[mv]
            e.N_virt = max(0, e.N_virt - 1)
            e.N += 1
            e.W += v

    if root.board.is_game_over():
        return {}, 0.0

    # Initial root expansion (single forward pass, never batched).
    root_priors, root_v = net.policy_value_batch(
        [root.board], device, histories=[history[:HISTORY_STEPS]]
    )[0]
    _expand_with(root, root_priors, root_v)
    _cache_put(_board_key(root.board), (root_priors, root_v))

    # lc0-style: apply policy softmax temperature at root to flatten priors.
    if policy_temp > 0.0 and abs(policy_temp - 1.0) > 1e-6 and root.edges:
        priors_arr = np.array([e.P for e in root.edges.values()], dtype=np.float64)
        priors_arr = np.power(priors_arr.clip(1e-12), 1.0 / policy_temp)
        priors_arr /= priors_arr.sum() + 1e-12
        for i, e in enumerate(root.edges.values()):
            e.P = float(priors_arr[i])

    if dirichlet_alpha > 0:
        moves = list(root.edges.keys())
        if moves:
            noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
            for i, mv in enumerate(moves):
                e = root.edges[mv]
                e.P = (1 - dirichlet_eps) * e.P + dirichlet_eps * float(noise[i])

    K = max(1, int(leaf_batch_size))
    sims_done = 0
    while sims_done < sims:
        # Collect up to K leaves needing network eval.
        pending: List[Tuple[List[Tuple[MCTSNode, chess.Move]], MCTSNode, list]] = []
        while len(pending) < K and (sims_done + len(pending)) < sims:
            path, leaf, leaf_hist = descend_to_leaf()

            # Terminal leaf: back up immediately, no network call needed.
            if leaf.board.is_game_over(claim_draw=False):
                leaf_v = _terminal_leaf_value(leaf.board)
                commit(path, leaf_v)
                sims_done += 1
                continue

            # Transposition hit: we know the eval for this position already.
            key = _board_key(leaf.board)
            cached = pv_cache.get(key)
            if cached is not None and not leaf.expanded:
                priors, v = cached
                _expand_with(leaf, priors, v)
                commit(path, v)
                sims_done += 1
                continue

            pending.append((path, leaf, leaf_hist))

        if not pending:
            continue

        leaves = [p[1].board for p in pending]
        histories = [p[2] for p in pending]
        results = net.policy_value_batch(leaves, device, histories=histories)

        for (path, leaf, _hist), (priors, v) in zip(pending, results):
            # Re-check in case a previous pending entry (same batch) already
            # expanded this node — avoid double expansion on transpositions.
            if not leaf.expanded:
                _expand_with(leaf, priors, v)
                _cache_put(_board_key(leaf.board), (priors, v))
            commit(path, v)
            sims_done += 1

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
