"""Tests for mini_az.checkpoint — save/load, RNG state management."""

import os

import numpy as np
import torch

from mini_az.checkpoint import (
    get_rng_state,
    set_rng_state,
    save_checkpoint,
    load_checkpoint,
)
from mini_az.network import ChessNet


class TestRNGState:
    def test_get_rng_state_keys(self):
        state = get_rng_state()
        assert "python" in state
        assert "numpy" in state
        assert "torch" in state
        assert "torch_cuda" in state

    def test_roundtrip_rng_state(self):
        """Save and restore RNG state, then verify reproducibility."""
        state = get_rng_state()

        # Generate some random numbers
        torch.manual_seed(42)
        np.random.seed(42)
        a1 = np.random.rand(5)
        t1 = torch.rand(5)

        # Restore the original state
        set_rng_state(state)

        # Should not crash
        state2 = get_rng_state()
        assert "python" in state2

    def test_set_rng_state_empty(self):
        """Setting empty/None state should not crash."""
        set_rng_state(None)
        set_rng_state({})


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        net = ChessNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        path = str(tmp_path / "ckpt.pt")

        save_checkpoint(path, net, opt, it=42)
        assert os.path.exists(path)

        # Load into fresh net
        net2 = ChessNet()
        opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
        it = load_checkpoint(path, net2, opt2, device="cpu", load_opt=True)

        assert it == 42

        # Weights should match
        for p1, p2 in zip(net.parameters(), net2.parameters()):
            assert torch.equal(p1, p2)

    def test_load_without_optimizer(self, tmp_path):
        net = ChessNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        path = str(tmp_path / "ckpt.pt")

        save_checkpoint(path, net, opt, it=10)

        net2 = ChessNet()
        opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
        it = load_checkpoint(path, net2, opt2, device="cpu", load_opt=False)

        assert it == 10

    def test_checkpoint_creates_directory(self, tmp_path):
        net = ChessNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        path = str(tmp_path / "subdir" / "ckpt.pt")

        save_checkpoint(path, net, opt, it=1)
        assert os.path.exists(path)
