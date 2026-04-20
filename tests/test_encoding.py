"""Tests for mini_az.encoding — board encoding, move encoding, CSV logging."""

import os
import tempfile

import chess
import numpy as np
import torch

from mini_az.encoding import (
    INPUT_PLANES,
    PLANES_PER_POS,
    AUX_PLANES,
    HISTORY_STEPS,
    PIECE_PLANES,
    PROMO_MAP,
    INV_PROMO_MAP,
    PL_MY_OO,
    PL_MY_OOO,
    PL_OPP_OO,
    PL_OPP_OOO,
    PL_EP,
    PL_HALFMOVE,
    PL_FULLMOVE,
    PL_CHECK,
    PL_REPETITION,
    board_to_tensor,
    encode_move_canonical,
    decode_move_from_canonical,
    legal_moves_canonical,
    append_eval_csv,
)


def _encode_pieces_legacy(planes, board, turn, plane_offset=0):
    """Reference per-piece Python-loop encoder (pre-vectorization baseline)."""
    for sq, piece in board.piece_map().items():
        ptype = piece.piece_type
        color = piece.color
        if turn == chess.WHITE:
            csq = sq
            my = (color == chess.WHITE)
        else:
            csq = chess.square_mirror(sq)
            my = (color == chess.BLACK)
        r = chess.square_rank(csq)
        f = chess.square_file(csq)
        base = PIECE_PLANES[ptype]
        idx = plane_offset + (base if my else base + 6)
        planes[idx, r, f] = 1.0


class TestConstants:
    def test_input_planes_count(self):
        assert INPUT_PLANES == 45

    def test_planes_per_pos(self):
        assert PLANES_PER_POS == 12

    def test_aux_planes(self):
        assert AUX_PLANES == 9

    def test_history_steps(self):
        assert HISTORY_STEPS == 2

    def test_plane_layout(self):
        # Piece planes: 12 * 3 = 36, then 9 aux = 45
        assert PLANES_PER_POS * (1 + HISTORY_STEPS) + AUX_PLANES == INPUT_PLANES


class TestPromoMap:
    def test_none_maps_to_zero(self):
        assert PROMO_MAP[None] == 0

    def test_roundtrip(self):
        for piece, idx in PROMO_MAP.items():
            assert INV_PROMO_MAP[idx] == piece

    def test_all_promos_covered(self):
        assert set(PROMO_MAP.keys()) == {None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}


class TestBoardToTensor:
    def test_starting_position_shape(self):
        board = chess.Board()
        t = board_to_tensor(board)
        assert t.shape == (INPUT_PLANES, 8, 8)
        assert t.dtype == torch.float32

    def test_starting_position_castling(self):
        board = chess.Board()
        t = board_to_tensor(board)
        # White to move: my = white, opp = black
        assert t[PL_MY_OO].sum() > 0, "White should have kingside castling"
        assert t[PL_MY_OOO].sum() > 0, "White should have queenside castling"
        assert t[PL_OPP_OO].sum() > 0, "Black should have kingside castling"
        assert t[PL_OPP_OOO].sum() > 0, "Black should have queenside castling"

    def test_no_ep_at_start(self):
        board = chess.Board()
        t = board_to_tensor(board)
        assert t[PL_EP].sum() == 0.0

    def test_ep_after_e4(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        # Now it's black's turn; e3 is EP square
        t = board_to_tensor(board)
        assert t[PL_EP].sum() > 0.0

    def test_halfmove_clock(self):
        board = chess.Board()
        t = board_to_tensor(board)
        assert t[PL_HALFMOVE, 0, 0] == 0.0

        # Advance without pawn move or capture
        board = chess.Board(fen="4k3/8/8/8/8/8/8/4K3 w - - 25 50")
        t = board_to_tensor(board)
        assert abs(t[PL_HALFMOVE, 0, 0].item() - 25.0 / 50.0) < 1e-6

    def test_fullmove_counter(self):
        board = chess.Board(fen="4k3/8/8/8/8/8/8/4K3 w - - 0 100")
        t = board_to_tensor(board)
        assert abs(t[PL_FULLMOVE, 0, 0].item() - 100.0 / 200.0) < 1e-6

    def test_check_plane(self):
        # Position with white in check
        board = chess.Board(fen="4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
        t = board_to_tensor(board)
        if board.is_check():
            assert t[PL_CHECK].sum() > 0.0

    def test_history_adds_planes(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        history = [chess.Board()]  # Starting position as history
        t = board_to_tensor(board, history=history)
        # History plane slot should have some piece data
        assert t[PLANES_PER_POS:2 * PLANES_PER_POS].sum() > 0.0

    def test_no_history_zeros(self):
        board = chess.Board()
        t = board_to_tensor(board, history=None)
        # History slots should be zero
        assert t[PLANES_PER_POS:2 * PLANES_PER_POS].sum() == 0.0
        assert t[2 * PLANES_PER_POS:3 * PLANES_PER_POS].sum() == 0.0

    def test_pieces_present_at_start(self):
        board = chess.Board()
        t = board_to_tensor(board)
        # There should be pieces in the current position planes (0-11)
        assert t[:PLANES_PER_POS].sum() > 0.0

    def test_repetition_plane_default(self):
        board = chess.Board()
        t = board_to_tensor(board)
        assert t[PL_REPETITION].sum() == 0.0


class TestEncodePiecesRegression:
    """Vectorized _encode_pieces must be bit-identical to the legacy loop."""

    def _random_positions(self, n: int, seed: int = 0xC0FFEE):
        import random
        rng = random.Random(seed)
        boards = [chess.Board()]
        while len(boards) < n:
            b = boards[-1]
            if b.is_game_over(claim_draw=False) or b.ply() > 120:
                boards.append(chess.Board())
                continue
            legals = list(b.legal_moves)
            if not legals:
                boards.append(chess.Board())
                continue
            nb = b.copy()
            nb.push(rng.choice(legals))
            boards.append(nb)
        return boards

    def test_vectorized_matches_legacy_for_random_positions(self):
        from mini_az.encoding import _encode_pieces  # vectorized
        positions = self._random_positions(500)

        for b in positions:
            for turn in (chess.WHITE, chess.BLACK):
                expected = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
                actual = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)

                _encode_pieces_legacy(expected, b, turn, plane_offset=0)
                _encode_pieces(actual, b, turn, plane_offset=0)

                assert np.array_equal(expected, actual), (
                    f"mismatch for fen={b.fen()!r} turn={turn}"
                )

    def test_vectorized_matches_legacy_with_offset(self):
        from mini_az.encoding import _encode_pieces
        positions = self._random_positions(50, seed=1234)

        for b in positions:
            for turn in (chess.WHITE, chess.BLACK):
                for offset in (0, 12, 24):
                    expected = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
                    actual = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
                    _encode_pieces_legacy(expected, b, turn, plane_offset=offset)
                    _encode_pieces(actual, b, turn, plane_offset=offset)
                    assert np.array_equal(expected, actual), (
                        f"mismatch for fen={b.fen()!r} turn={turn} offset={offset}"
                    )

    def test_board_to_tensor_matches_legacy_full(self):
        """End-to-end: board_to_tensor with history must be identical."""
        from mini_az.encoding import _encode_pieces
        positions = self._random_positions(200, seed=42)

        for i, b in enumerate(positions):
            hist = [positions[i - 1]] if i >= 1 else []
            if i >= 2:
                hist = [positions[i - 1], positions[i - 2]]
            t_now = board_to_tensor(b, history=hist).numpy()

            # Reconstruct expected using legacy encoder
            expected = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
            _encode_pieces_legacy(expected, b, b.turn, plane_offset=0)
            for step in range(HISTORY_STEPS):
                off = PLANES_PER_POS * (1 + step)
                if step < len(hist):
                    _encode_pieces_legacy(expected, hist[step], b.turn, plane_offset=off)

            # Piece planes (0..35) must match exactly; aux planes come from board_to_tensor
            # so we only compare the piece portion here.
            assert np.array_equal(
                t_now[: PLANES_PER_POS * (1 + HISTORY_STEPS)],
                expected[: PLANES_PER_POS * (1 + HISTORY_STEPS)],
            ), f"piece planes mismatch for fen={b.fen()!r}"


class TestMoveEncoding:
    def test_e2e4_white(self):
        board = chess.Board()
        mv = chess.Move.from_uci("e2e4")
        fs, ts, p = encode_move_canonical(board, mv)
        assert fs == chess.E2
        assert ts == chess.E4
        assert p == 0

    def test_e7e5_black(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        mv = chess.Move.from_uci("e7e5")
        fs, ts, p = encode_move_canonical(board, mv)
        # Black: squares are mirrored
        assert p == 0

    def test_promotion(self):
        board = chess.Board(fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        mv = chess.Move.from_uci("a7a8q")
        fs, ts, p = encode_move_canonical(board, mv)
        assert p == PROMO_MAP[chess.QUEEN]

    def test_roundtrip_encode_decode(self):
        board = chess.Board()
        for mv in list(board.legal_moves)[:10]:
            fs, ts, p = encode_move_canonical(board, mv)
            decoded = decode_move_from_canonical(board, fs, ts, p)
            assert decoded == mv, f"Roundtrip failed: {mv} → ({fs},{ts},{p}) → {decoded}"


class TestLegalMovesCanonical:
    def test_starting_position(self):
        board = chess.Board()
        moves = legal_moves_canonical(board)
        assert len(moves) == 20  # 16 pawn + 4 knight moves

    def test_tuple_structure(self):
        board = chess.Board()
        moves = legal_moves_canonical(board)
        for fs, ts, p, mv in moves:
            assert isinstance(fs, int)
            assert isinstance(ts, int)
            assert isinstance(p, int)
            assert isinstance(mv, chess.Move)
            assert 0 <= fs < 64
            assert 0 <= ts < 64
            assert 0 <= p <= 4


class TestAppendEvalCsv:
    def test_create_new_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.csv")
            append_eval_csv(path, {"iter": 1, "score": 0.5})
            assert os.path.exists(path)

    def test_append_to_existing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.csv")
            append_eval_csv(path, {"iter": 1, "score": 0.5})
            append_eval_csv(path, {"iter": 2, "score": 0.6})
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows

    def test_append_with_new_column(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.csv")
            append_eval_csv(path, {"iter": 1, "score": 0.5})
            append_eval_csv(path, {"iter": 2, "score": 0.6, "extra": 42})
            with open(path) as f:
                lines = f.readlines()
            assert "extra" in lines[0]  # New column in header
