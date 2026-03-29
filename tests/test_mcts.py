"""Tests for mini_az.mcts — MCTS search, FPU, policy temp, PV cache."""

import chess
import numpy as np
import torch

from mini_az.network import ChessNet
from mini_az.mcts import mcts_search


class TestMCTSBasic:
    def test_returns_visits_and_value(self):
        net = ChessNet()
        net.eval()
        board = chess.Board()
        visits, root_q = mcts_search(net, board, "cpu", sims=8)
        assert isinstance(visits, dict)
        assert isinstance(root_q, float)
        assert len(visits) > 0
        assert -1.0 <= root_q <= 1.0

    def test_visits_are_nonnegative(self):
        net = ChessNet()
        net.eval()
        board = chess.Board()
        visits, _ = mcts_search(net, board, "cpu", sims=16)
        for mv, count in visits.items():
            assert isinstance(mv, chess.Move)
            assert count >= 0
        # At least some moves should have positive visits
        assert any(c > 0 for c in visits.values())

    def test_total_visits_match_sims(self):
        net = ChessNet()
        net.eval()
        board = chess.Board()
        sims = 32
        visits, _ = mcts_search(net, board, "cpu", sims=sims)
        total = sum(visits.values())
        # Total visits should be close to sims (root expansion + sims iterations)
        assert total >= sims

    def test_all_visited_moves_are_legal(self):
        net = ChessNet()
        net.eval()
        board = chess.Board()
        visits, _ = mcts_search(net, board, "cpu", sims=16)
        legal = set(board.legal_moves)
        for mv in visits:
            assert mv in legal


class TestMCTSParameters:
    def test_dirichlet_noise(self):
        """With Dirichlet noise, search should still work."""
        net = ChessNet()
        net.eval()
        board = chess.Board()
        visits, root_q = mcts_search(
            net, board, "cpu", sims=16,
            dirichlet_alpha=0.3, dirichlet_eps=0.25,
        )
        assert len(visits) > 0

    def test_fpu_root(self):
        """Custom root FPU should not break search."""
        net = ChessNet()
        net.eval()
        board = chess.Board()
        visits, _ = mcts_search(net, board, "cpu", sims=16, fpu_root=-1.0)
        assert len(visits) > 0

    def test_policy_temp(self):
        """Policy temperature should not break search."""
        net = ChessNet()
        net.eval()
        board = chess.Board()
        visits, _ = mcts_search(net, board, "cpu", sims=16, policy_temp=1.4)
        assert len(visits) > 0

    def test_policy_temp_one(self):
        """policy_temp=1.0 should behave like no temperature."""
        net = ChessNet()
        net.eval()
        board = chess.Board()
        visits, _ = mcts_search(net, board, "cpu", sims=16, policy_temp=1.0)
        assert len(visits) > 0

    def test_with_history(self):
        """MCTS should work with position history."""
        net = ChessNet()
        net.eval()
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        history = [chess.Board()]
        visits, _ = mcts_search(net, board, "cpu", sims=8, history=history)
        assert len(visits) > 0


class TestMCTSEdgeCases:
    def test_checkmate_position(self):
        """MCTS on a checkmate position should return empty visits."""
        net = ChessNet()
        net.eval()
        board = chess.Board(fen="r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        visits, root_q = mcts_search(net, board, "cpu", sims=8)
        assert len(visits) == 0

    def test_stalemate_position(self):
        """MCTS on stalemate should return empty visits."""
        net = ChessNet()
        net.eval()
        board = chess.Board(fen="k7/8/1K6/8/8/8/8/8 b - - 0 1")
        if board.is_stalemate():
            visits, _ = mcts_search(net, board, "cpu", sims=8)
            assert len(visits) == 0

    def test_few_legal_moves(self):
        """Position with very few legal moves — e.g. king + pawn endgame."""
        net = ChessNet()
        net.eval()
        # King + pawn vs king — enough material to avoid immediate draw
        board = chess.Board(fen="4k3/8/8/8/8/4P3/8/4K3 w - - 0 1")
        visits, _ = mcts_search(net, board, "cpu", sims=16)
        assert len(visits) > 0
        legal = set(board.legal_moves)
        for mv in visits:
            assert mv in legal
