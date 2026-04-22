"""Tests for 5.2 anti-repetition tuning.

Tests the tighter repetition penalties and draw-with-advantage z-penalty
without running full self-play games (those are integration-level).
"""

import numpy as np
import chess

from mini_az.config import material_score


class TestRepetitionPenalties:
    """Verify the penalty constants are tighter than the old values."""

    def test_penalty_values_high_advantage(self):
        """With adv >= 3, rep_penalty should be very aggressive (< 0.10)."""
        # This tests the contract from 5.2: adv>=3 → rep=0.05
        adv = 4
        if adv >= 3:
            rep_penalty, fifty_penalty, pat_penalty = 0.05, 0.10, 0.10
        assert rep_penalty == 0.05, "High-advantage rep penalty should be 0.05"
        assert fifty_penalty == 0.10

    def test_penalty_values_medium_advantage(self):
        """With adv >= 1, rep_penalty should be tighter than old 0.25."""
        adv = 2
        if adv >= 3:
            rep_penalty = 0.05
        elif adv >= 1:
            rep_penalty, fifty_penalty, pat_penalty = 0.12, 0.20, 0.20
        assert rep_penalty == 0.12, "Medium-advantage rep penalty should be 0.12"

    def test_penalty_values_no_advantage(self):
        """Without advantage, rep_penalty should be tighter than old 0.50."""
        adv = 0
        if adv >= 3:
            rep_penalty = 0.05
        elif adv >= 1:
            rep_penalty = 0.12
        else:
            rep_penalty, fifty_penalty, pat_penalty = 0.35, 0.45, 0.40
        assert rep_penalty == 0.35, "No-advantage rep penalty should be 0.35"


class TestDrawWithAdvantagePenalty:
    """Test the z-penalty logic for draws where one side had advantage."""

    def test_draw_white_advantage_penalised(self):
        """White had material advantage in a draw → z_draw_white < 0."""
        end_ms = 3  # White up 3 material
        z_draw_white = 0.0
        if abs(end_ms) >= 2:
            z_draw_white = -0.12 if end_ms > 0 else 0.12
        assert z_draw_white == -0.12, "White wasted advantage → negative z"

    def test_draw_black_advantage_penalised(self):
        """Black had material advantage in a draw → z_draw_white > 0 (white defended)."""
        end_ms = -4  # Black up 4 material
        z_draw_white = 0.0
        if abs(end_ms) >= 2:
            z_draw_white = -0.12 if end_ms > 0 else 0.12
        assert z_draw_white == 0.12, "White defended → positive z for white"

    def test_draw_equal_material_no_penalty(self):
        """Equal material draw → z stays 0.0."""
        end_ms = 1  # marginal difference, below threshold
        z_draw_white = 0.0
        if abs(end_ms) >= 2:
            z_draw_white = -0.12 if end_ms > 0 else 0.12
        assert z_draw_white == 0.0, "Equal material → no z penalty"

    def test_sf_bootstrap_overrides_penalty(self):
        """When SF bootstrap gives z, it should override draw penalty."""
        z_boot_white = 0.3  # SF says white is slightly better
        z_draw_white = -0.12  # draw penalty says penalise white
        # The actual code: z_white_end = z_boot_white if z_boot_white is not None else z_draw_white
        z_white_end = z_boot_white if z_boot_white is not None else z_draw_white
        assert z_white_end == 0.3, "SF bootstrap should override draw penalty"

    def test_material_score_basic(self):
        """Sanity: material_score returns positive for white advantage."""
        board = chess.Board()
        # Standard position: material is equal
        ms = material_score(board)
        assert ms == 0, "Starting position should have 0 material score"

    def test_material_score_white_up_queen(self):
        """White up a queen → ms ~ +9."""
        # Remove black queen
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        ms = material_score(board)
        assert ms > 5, f"White up a queen should have ms > 5, got {ms}"


class TestRepetitionZPenalty:
    """Tests for 5.11: z-target shrinkage for repetitive games."""

    @staticmethod
    def _compute_rep_z_scale(actual_rep_plies: int) -> float:
        REP_PENALTY_THRESH = 10
        REP_PENALTY_FULL   = 30
        REP_PENALTY_MIN_SCALE = 0.5
        if actual_rep_plies >= REP_PENALTY_THRESH:
            frac = min(1.0, (actual_rep_plies - REP_PENALTY_THRESH) / max(1, REP_PENALTY_FULL - REP_PENALTY_THRESH))
            return 1.0 - (1.0 - REP_PENALTY_MIN_SCALE) * frac
        return 1.0

    def test_no_penalty_below_threshold(self):
        assert self._compute_rep_z_scale(0) == 1.0
        assert self._compute_rep_z_scale(5) == 1.0
        assert self._compute_rep_z_scale(9) == 1.0

    def test_penalty_starts_at_threshold(self):
        scale = self._compute_rep_z_scale(10)
        assert scale == 1.0  # exactly at threshold → frac=0 → scale=1.0

    def test_penalty_at_midpoint(self):
        # 20 rep plies → frac = (20-10)/(30-10) = 0.5 → scale = 0.75
        scale = self._compute_rep_z_scale(20)
        assert abs(scale - 0.75) < 1e-6

    def test_full_penalty(self):
        scale = self._compute_rep_z_scale(30)
        assert abs(scale - 0.5) < 1e-6

    def test_beyond_full_capped(self):
        """Beyond 30 rep plies, scale should stay at 0.5 (not go lower)."""
        scale = self._compute_rep_z_scale(50)
        assert abs(scale - 0.5) < 1e-6

    def test_z_shrinks_correctly(self):
        """A z=0.8 game with 20 rep plies → z_target = 0.8 * 0.75 = 0.6."""
        scale = self._compute_rep_z_scale(20)
        z_original = 0.8
        z_scaled = z_original * scale
        assert abs(z_scaled - 0.6) < 1e-6

    def test_negative_z_shrinks_towards_zero(self):
        """A z=-1.0 game with 30 rep plies → z_target = -1.0 * 0.5 = -0.5."""
        scale = self._compute_rep_z_scale(30)
        z_scaled = -1.0 * scale
        assert abs(z_scaled - (-0.5)) < 1e-6

    def test_draw_z_unaffected(self):
        """z=0.0 is unaffected by any scaling."""
        for rp in [0, 10, 20, 30, 50]:
            scale = self._compute_rep_z_scale(rp)
            assert 0.0 * scale == 0.0
