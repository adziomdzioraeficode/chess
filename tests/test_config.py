"""Tests for mini_az.config — utility functions, opening book, logging."""

import chess

from mini_az.config import (
    apply_random_opening,
    material_score,
    OPENING_BOOK_UCI,
    NO_PROGRESS_HALFMOVE,
)


class TestMaterialScore:
    def test_starting_position(self):
        board = chess.Board()
        assert material_score(board) == 0

    def test_white_up_queen(self):
        # Remove black queen
        board = chess.Board()
        board.remove_piece_at(chess.D8)
        assert material_score(board) == 9

    def test_black_up_pawn(self):
        board = chess.Board()
        board.remove_piece_at(chess.E2)
        assert material_score(board) == -1

    def test_empty_board_with_kings(self):
        board = chess.Board(fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        assert material_score(board) == 0


class TestOpeningBook:
    def test_book_not_empty(self):
        assert len(OPENING_BOOK_UCI) > 0

    def test_book_lines_are_valid(self):
        for line in OPENING_BOOK_UCI:
            board = chess.Board()
            for uci in line:
                mv = chess.Move.from_uci(uci)
                assert mv in board.legal_moves, f"Illegal move {uci} in book line"
                board.push(mv)

    def test_apply_random_opening_zero_plies(self):
        board = chess.Board()
        played = apply_random_opening(board, 0)
        assert played == 0
        assert board.fen() == chess.STARTING_FEN

    def test_apply_random_opening_some_plies(self):
        board = chess.Board()
        played = apply_random_opening(board, 4)
        assert 0 <= played <= 4
        # Board should have advanced
        assert board.fullmove_number >= 1

    def test_apply_random_opening_large_max(self):
        board = chess.Board()
        played = apply_random_opening(board, 100)
        # Should be bounded by the book line length
        max_line = max(len(line) for line in OPENING_BOOK_UCI)
        assert played <= max_line


class TestConstants:
    def test_no_progress_halfmove(self):
        assert NO_PROGRESS_HALFMOVE == 80
