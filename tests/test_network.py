"""Tests for mini_az.network — ChessNet architecture, forward pass, WDL head."""

import chess
import numpy as np
import torch

from mini_az.network import ChessNet, SEBlock, ResidualBlock
from mini_az.encoding import INPUT_PLANES, board_to_tensor, legal_moves_canonical


class TestSEBlock:
    def test_output_shape(self):
        se = SEBlock(channels=16, ratio=4)
        x = torch.randn(2, 16, 8, 8)
        out = se(x)
        assert out.shape == (2, 16, 8, 8)


class TestResidualBlock:
    def test_output_shape(self):
        blk = ResidualBlock(channels=16, gn_groups=4)
        x = torch.randn(2, 16, 8, 8)
        out = blk(x)
        assert out.shape == (2, 16, 8, 8)

    def test_residual_connection(self):
        """Output should differ from zero (residual + SE + ReLU)."""
        blk = ResidualBlock(channels=16, gn_groups=4)
        x = torch.ones(1, 16, 8, 8)
        out = blk(x)
        assert out.sum() != 0.0


class TestChessNet:
    def test_default_creation(self):
        net = ChessNet()
        params = sum(p.numel() for p in net.parameters())
        assert params > 1_000_000  # Should be ~5.5M

    def test_encode_board_shape(self):
        net = ChessNet()
        x = torch.randn(2, INPUT_PLANES, 8, 8)
        emb = net.encode_board(x)
        assert emb.shape == (2, 512)  # board_dim default

    def test_forward_policy_value_shapes(self):
        net = ChessNet()
        net.eval()
        B, L = 2, 5
        boards = torch.randn(B, INPUT_PLANES, 8, 8)
        fs = torch.randint(0, 64, (B, L))
        ts = torch.randint(0, 64, (B, L))
        pr = torch.randint(0, 5, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)

        logits, wdl_logits = net.forward_policy_value(boards, fs, ts, pr, mask)
        assert logits.shape == (B, L)
        assert wdl_logits.shape == (B, 3)

    def test_masked_moves_are_neg_inf(self):
        net = ChessNet()
        net.eval()
        B, L = 1, 5
        boards = torch.randn(B, INPUT_PLANES, 8, 8)
        fs = torch.randint(0, 64, (B, L))
        ts = torch.randint(0, 64, (B, L))
        pr = torch.zeros(B, L, dtype=torch.long)
        mask = torch.tensor([[True, True, False, False, False]])

        logits, _ = net.forward_policy_value(boards, fs, ts, pr, mask)
        assert logits[0, 2].item() == float("-inf")
        assert logits[0, 0].item() != float("-inf")

    def test_wdl_sums_to_one(self):
        net = ChessNet()
        net.eval()
        B, L = 1, 3
        boards = torch.randn(B, INPUT_PLANES, 8, 8)
        fs = torch.randint(0, 64, (B, L))
        ts = torch.randint(0, 64, (B, L))
        pr = torch.zeros(B, L, dtype=torch.long)
        mask = torch.ones(B, L, dtype=torch.bool)

        _, wdl_logits = net.forward_policy_value(boards, fs, ts, pr, mask)
        wdl_probs = torch.softmax(wdl_logits, dim=-1)
        assert abs(wdl_probs.sum().item() - 1.0) < 1e-5


class TestPolicyValueSingle:
    def test_starting_position(self):
        net = ChessNet()
        net.eval()
        board = chess.Board()
        priors, v = net.policy_value_single(board, "cpu")
        assert len(priors) == 20  # 20 legal moves at start
        assert abs(sum(priors.values()) - 1.0) < 1e-5
        assert -1.0 <= v <= 1.0

    def test_after_moves(self):
        net = ChessNet()
        net.eval()
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))
        priors, v = net.policy_value_single(board, "cpu")
        assert len(priors) == len(list(board.legal_moves))
        assert abs(sum(priors.values()) - 1.0) < 1e-5

    def test_with_history(self):
        net = ChessNet()
        net.eval()
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        history = [chess.Board()]
        priors, v = net.policy_value_single(board, "cpu", history=history)
        assert len(priors) > 0
        assert -1.0 <= v <= 1.0

    def test_no_legal_moves(self):
        """Checkmate position should return empty priors."""
        net = ChessNet()
        net.eval()
        # Scholar's mate position (black is checkmated)
        board = chess.Board(fen="r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        priors, v = net.policy_value_single(board, "cpu")
        assert len(priors) == 0
        assert v == 0.0
