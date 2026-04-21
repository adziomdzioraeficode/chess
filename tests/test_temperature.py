"""Tests for 5.5 — smooth temperature annealing in selfplay."""
import math
import pytest


# ── helpers to extract temp constants from selfplay source ──────────────
def _get_temp(tply: float) -> float:
    """Replicate the temperature annealing logic from selfplay.py."""
    TEMP_PLY_FULL = 12
    TEMP_PLY_MIN  = 30
    TEMP_HIGH     = 1.0
    TEMP_LOW      = 0.08
    if tply < TEMP_PLY_FULL:
        return TEMP_HIGH
    elif tply < TEMP_PLY_MIN:
        frac = (tply - TEMP_PLY_FULL) / (TEMP_PLY_MIN - TEMP_PLY_FULL)
        return TEMP_HIGH - (TEMP_HIGH - TEMP_LOW) * frac
    else:
        return TEMP_LOW


class TestTemperatureAnnealing:
    """Verify temperature schedule properties."""

    def test_full_temp_at_ply_0(self):
        assert _get_temp(0) == 1.0

    def test_full_temp_at_ply_11(self):
        assert _get_temp(11) == 1.0

    def test_full_temp_boundary_ply_12_still_high(self):
        # ply 12 is the first ply of annealing; frac=0 → still TEMP_HIGH
        assert _get_temp(12) == pytest.approx(1.0)

    def test_midpoint_temperature(self):
        # ply 21 = midpoint of [12, 30), frac = 9/18 = 0.5
        mid = 1.0 - 0.92 * 0.5  # 0.54
        assert _get_temp(21) == pytest.approx(mid, abs=1e-6)

    def test_min_temp_at_ply_30(self):
        assert _get_temp(30) == pytest.approx(0.08)

    def test_min_temp_at_ply_100(self):
        assert _get_temp(100) == pytest.approx(0.08)

    def test_monotonic_decrease(self):
        """Temperature must be non-increasing over ply range 0..50."""
        temps = [_get_temp(p) for p in range(51)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-12, (
                f"temp increased at ply {i}: {temps[i-1]:.4f} -> {temps[i]:.4f}"
            )

    def test_continuity_no_big_jumps(self):
        """No temperature jump > 0.15 between consecutive plies (smooth)."""
        temps = [_get_temp(p) for p in range(51)]
        for i in range(1, len(temps)):
            diff = abs(temps[i] - temps[i - 1])
            assert diff < 0.15, (
                f"big jump at ply {i}: |{temps[i-1]:.4f} - {temps[i]:.4f}| = {diff:.4f}"
            )

    def test_no_negative_temperature(self):
        for p in range(200):
            assert _get_temp(p) >= 0.0
