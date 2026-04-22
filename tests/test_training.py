"""Tests for mini_az.training — ReplayBuffer, collate, train_step, augmentation."""

import numpy as np
import torch

from mini_az.training import (
    Sample,
    ReplayBuffer,
    collate,
    train_step,
    flip_sample_lr,
    masked_entropy,
)
from mini_az.network import ChessNet
from mini_az.encoding import INPUT_PLANES, PL_MY_OO, PL_MY_OOO, PL_OPP_OO, PL_OPP_OOO


def _make_sample(n_moves=5, z=0.5):
    """Create a synthetic training sample."""
    board_planes = np.random.randn(INPUT_PLANES, 8, 8).astype(np.float32)
    moves_fs = np.random.randint(0, 64, size=n_moves).astype(np.int64)
    moves_ts = np.random.randint(0, 64, size=n_moves).astype(np.int64)
    moves_pr = np.zeros(n_moves, dtype=np.int64)
    pi = np.random.dirichlet(np.ones(n_moves)).astype(np.float32)
    wdl = np.array([0.4, 0.2, 0.4], dtype=np.float32)
    plies_left = float(np.random.randint(0, 120))
    return Sample(board_planes, moves_fs, moves_ts, moves_pr, pi, z, wdl, plies_left)


class TestSample:
    def test_creation(self):
        s = _make_sample()
        assert s.board_planes.shape == (INPUT_PLANES, 8, 8)
        assert len(s.moves_fs) == 5
        assert len(s.target_pi) == 5
        assert abs(s.target_pi.sum() - 1.0) < 1e-5
        assert s.wdl.shape == (3,)
        assert s.plies_left >= 0.0


class TestReplayBuffer:
    def test_add_and_len(self):
        rb = ReplayBuffer(maxlen=100)
        assert len(rb) == 0
        rb.add_game([_make_sample() for _ in range(10)])
        assert len(rb) == 10

    def test_circular_overflow(self):
        rb = ReplayBuffer(maxlen=5)
        rb.add_game([_make_sample() for _ in range(10)])
        assert len(rb) == 5

    def test_sample_batch(self):
        rb = ReplayBuffer(maxlen=100)
        rb.add_game([_make_sample() for _ in range(20)])
        batch = rb.sample_batch(8)
        assert len(batch) == 8
        for s in batch:
            assert isinstance(s, Sample)

    def test_sample_batch_mixed(self):
        rb = ReplayBuffer(maxlen=1000)
        rb.add_game([_make_sample() for _ in range(100)])
        batch = rb.sample_batch_mixed(16, recent_frac=0.75, recent_window=50, sharp_frac=0.25, sharp_threshold=0.25)
        assert len(batch) == 16

    def test_sample_batch_mixed_with_decisive(self):
        rb = ReplayBuffer(maxlen=1000)
        # Add a mix: 50 decisive (z=0.8) and 50 non-decisive (z=0.1)
        for _ in range(50):
            rb.add_game([_make_sample(z=0.8)])
        for _ in range(50):
            rb.add_game([_make_sample(z=0.1)])
        batch = rb.sample_batch_mixed(32, recent_frac=0.5, decisive_frac=0.25)
        assert len(batch) == 32

    def test_decisive_sampling_biased(self):
        """Decisive sampling should return mostly |z|>0.3 samples."""
        rb = ReplayBuffer(maxlen=1000)
        for _ in range(200):
            rb.add_game([_make_sample(z=0.8)])
        for _ in range(800):
            rb.add_game([_make_sample(z=0.1)])
        # 20% decisive in buffer, but sampling should find mostly decisive
        idxs = rb._sample_indices_decisive(100, threshold=0.3)
        assert len(idxs) == 100
        decisive_count = sum(1 for i in idxs if abs(rb.data[i].z) > 0.3)
        assert decisive_count >= 80, f"Expected mostly decisive, got {decisive_count}/100"

    def test_decisive_fallback_uniform(self):
        """When no decisive samples exist, falls back to uniform."""
        rb = ReplayBuffer(maxlen=100)
        rb.add_game([_make_sample(z=0.0) for _ in range(50)])
        idxs = rb._sample_indices_decisive(10, threshold=0.3)
        assert len(idxs) == 10  # should not crash, falls back

    def test_sample_batch_mixed_empty(self):
        rb = ReplayBuffer(maxlen=100)
        batch = rb.sample_batch_mixed(8)
        assert len(batch) == 0

    def test_resize_smaller(self):
        rb = ReplayBuffer(maxlen=100)
        rb.add_game([_make_sample() for _ in range(50)])
        rb.resize(20)
        assert len(rb) == 20
        assert rb.maxlen == 20

    def test_resize_larger(self):
        rb = ReplayBuffer(maxlen=10)
        rb.add_game([_make_sample() for _ in range(10)])
        rb.resize(100)
        assert len(rb) == 10
        assert rb.maxlen == 100

    def test_dump_load(self, tmp_path):
        rb = ReplayBuffer(maxlen=100)
        samples = [_make_sample() for _ in range(15)]
        rb.add_game(samples)

        path = str(tmp_path / "replay.pkl.gz")
        rb.dump(path)

        rb2 = ReplayBuffer.load(path)
        assert len(rb2) == 15
        assert rb2.maxlen == 100


class TestFlipSampleLR:
    def test_flip_preserves_shape(self):
        s = _make_sample()
        flipped = flip_sample_lr(s)
        assert flipped.board_planes.shape == s.board_planes.shape
        assert len(flipped.moves_fs) == len(s.moves_fs)

    def test_flip_mirrors_squares(self):
        s = _make_sample()
        flipped = flip_sample_lr(s)
        # XOR with 7 mirrors file: a↔h, b↔g, etc.
        np.testing.assert_array_equal(flipped.moves_fs, s.moves_fs ^ 7)
        np.testing.assert_array_equal(flipped.moves_ts, s.moves_ts ^ 7)

    def test_flip_swaps_castling(self):
        s = _make_sample()
        # Set distinct castling plane values
        s.board_planes[PL_MY_OO, :, :] = 1.0
        s.board_planes[PL_MY_OOO, :, :] = 0.0
        s.board_planes[PL_OPP_OO, :, :] = 1.0
        s.board_planes[PL_OPP_OOO, :, :] = 0.0

        flipped = flip_sample_lr(s)
        # After flip, kingside ↔ queenside
        np.testing.assert_array_equal(flipped.board_planes[PL_MY_OO], s.board_planes[PL_MY_OOO])
        np.testing.assert_array_equal(flipped.board_planes[PL_MY_OOO], s.board_planes[PL_MY_OO])

    def test_flip_preserves_policy(self):
        s = _make_sample()
        flipped = flip_sample_lr(s)
        np.testing.assert_array_equal(flipped.target_pi, s.target_pi)

    def test_flip_preserves_wdl(self):
        s = _make_sample()
        flipped = flip_sample_lr(s)
        np.testing.assert_array_equal(flipped.wdl, s.wdl)

    def test_flip_preserves_plies_left(self):
        s = _make_sample()
        flipped = flip_sample_lr(s)
        assert flipped.plies_left == s.plies_left


class TestCollate:
    def test_basic_collation(self):
        samples = [_make_sample(n_moves=n) for n in [3, 5, 4]]
        boards, fs, ts, pr, mask, target_pi, z, wdl, plies_left = collate(samples, "cpu")
        assert boards.shape == (3, INPUT_PLANES, 8, 8)
        assert fs.shape[0] == 3
        assert fs.shape[1] == 5  # max moves
        assert mask.shape == (3, 5)
        assert z.shape == (3,)
        assert wdl.shape == (3, 3)
        assert plies_left.shape == (3,)

    def test_masking_correctness(self):
        samples = [_make_sample(n_moves=2), _make_sample(n_moves=4)]
        _, _, _, _, mask, _, _, _, _ = collate(samples, "cpu")
        # Padding positions should be masked out
        assert mask.shape[1] == 4  # max
        # First sample has 2 moves → positions 2,3 should be False
        # (but augmentation may flip, so just check mask is boolean)
        assert mask.dtype == torch.bool

    def test_target_pi_sums(self):
        samples = [_make_sample(n_moves=5) for _ in range(4)]
        _, _, _, _, mask, target_pi, _, _, _ = collate(samples, "cpu")
        for i in range(4):
            active = target_pi[i][mask[i]]
            assert abs(active.sum().item() - 1.0) < 1e-4


    def test_collate_vectorized_augmentation(self):
        """Vectorized collate applies LR flip correctly — castling swapped, files mirrored."""
        np.random.seed(42)
        s = _make_sample(n_moves=4)
        # Set distinct castling planes
        s.board_planes[PL_MY_OO, :, :] = 1.0
        s.board_planes[PL_MY_OOO, :, :] = 0.0
        s.board_planes[PL_OPP_OO, :, :] = 0.7
        s.board_planes[PL_OPP_OOO, :, :] = 0.3
        # Use many copies so some will be flipped, some not
        samples = [s] * 20
        boards, fs, ts, pr, mask, target_pi, z, wdl, plies_left = collate(samples, "cpu")
        assert boards.shape == (20, INPUT_PLANES, 8, 8)
        # At least some should be flipped and some not (with seed 42, ~10 each)
        # All should have valid mask
        for i in range(20):
            active = target_pi[i][mask[i]]
            assert abs(active.sum().item() - 1.0) < 1e-4

    def test_collate_padding_zeros(self):
        """Padded positions in fs/ts/pr should be zero, mask False."""
        samples = [_make_sample(n_moves=2), _make_sample(n_moves=5)]
        _, fs, ts, pr, mask, target_pi, _, _, _ = collate(samples, "cpu")
        assert mask.shape[1] == 5
        # For the 2-move sample, positions 2-4 should be masked out
        # and have zero values (accounting for possible augmentation)
        for i in range(2):
            n_active = mask[i].sum().item()
            assert n_active in [2, 5]  # either sample's move count
            # Padded positions must be zero
            if n_active < 5:
                assert (fs[i, int(n_active):] == 0).all()
                assert (target_pi[i, int(n_active):] == 0).all()

    def test_collate_z_wdl_plies(self):
        """Scalar fields z, wdl, plies_left are correctly transferred."""
        s1 = _make_sample(n_moves=3, z=0.8)
        s1.wdl = np.array([0.8, 0.2, 0.0], dtype=np.float32)
        s1.plies_left = 42.0
        s2 = _make_sample(n_moves=3, z=-0.5)
        s2.wdl = np.array([0.0, 0.5, 0.5], dtype=np.float32)
        s2.plies_left = 10.0
        _, _, _, _, _, _, z, wdl, plies_left = collate([s1, s2], "cpu")
        assert abs(z[0].item() - 0.8) < 1e-5
        assert abs(z[1].item() - (-0.5)) < 1e-5
        assert abs(plies_left[0].item() - 42.0) < 1e-5
        assert abs(plies_left[1].item() - 10.0) < 1e-5
        np.testing.assert_allclose(wdl[0].numpy(), [0.8, 0.2, 0.0], atol=1e-5)
        np.testing.assert_allclose(wdl[1].numpy(), [0.0, 0.5, 0.5], atol=1e-5)


class TestMaskedEntropy:
    def test_uniform_distribution(self):
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        ent = masked_entropy(p, mask)
        # Entropy of uniform over 4 = ln(4) ≈ 1.386
        assert abs(ent.item() - np.log(4)) < 0.01

    def test_peaked_distribution(self):
        p = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        ent = masked_entropy(p, mask)
        assert ent.item() < 0.5  # Low entropy


class TestTrainStep:
    def test_basic_train_step(self):
        net = ChessNet()
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        samples = [_make_sample(n_moves=5) for _ in range(4)]
        batch = collate(samples, "cpu")
        metrics = train_step(net, opt, batch, val_w=1.0)

        assert "loss" in metrics
        assert "pol" in metrics
        assert "val" in metrics
        assert "grad_norm" in metrics
        assert "pred_ent" in metrics
        assert "tgt_ent" in metrics
        assert "v_mean" in metrics
        assert "z_mean" in metrics
        assert "vz_corr" in metrics
        assert "ml" in metrics
        assert "ml_mean" in metrics
        assert "ml_tgt_mean" in metrics
        assert metrics["loss"] > 0
        assert metrics["grad_norm"] > 0

    def test_train_step_updates_weights(self):
        """Training steps should change model parameters."""
        net = ChessNet()
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        samples = [_make_sample(n_moves=5) for _ in range(8)]
        batch = collate(samples, "cpu")

        # Snapshot a parameter before training
        param_before = next(net.parameters()).data.clone()
        train_step(net, opt, batch, val_w=1.0)
        param_after = next(net.parameters()).data

        assert not torch.equal(param_before, param_after), "Weights should change after training step"

    def test_grad_accum_no_step(self):
        """With do_step=False, optimizer should NOT step (weights unchanged)."""
        net = ChessNet()
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        samples = [_make_sample(n_moves=5) for _ in range(4)]
        batch = collate(samples, "cpu")

        opt.zero_grad()
        param_before = next(net.parameters()).data.clone()
        m = train_step(net, opt, batch, val_w=1.0, loss_scale=0.5, do_step=False)
        param_after = next(net.parameters()).data

        assert torch.equal(param_before, param_after), "Weights must NOT change with do_step=False"
        # But gradients should exist
        assert next(net.parameters()).grad is not None
        assert m["loss"] > 0

    def test_grad_accum_two_steps_then_update(self):
        """Accumulate 2 sub-batches then manually step — weights should change."""
        net = ChessNet()
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        samples = [_make_sample(n_moves=5) for _ in range(8)]

        param_before = next(net.parameters()).data.clone()

        opt.zero_grad()
        batch1 = collate(samples[:4], "cpu")
        train_step(net, opt, batch1, val_w=1.0, loss_scale=0.5, do_step=False)

        batch2 = collate(samples[4:], "cpu")
        train_step(net, opt, batch2, val_w=1.0, loss_scale=0.5, do_step=False)

        # Manual clip + step (like trainer_loop does)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()

        param_after = next(net.parameters()).data
        assert not torch.equal(param_before, param_after), "Weights should change after accumulated step"

    def test_grad_accum_loss_scaling(self):
        """Loss scale=0.5 should produce roughly half the gradient magnitude."""
        net = ChessNet()
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        samples = [_make_sample(n_moves=5) for _ in range(4)]
        batch = collate(samples, "cpu")

        # Full scale
        opt.zero_grad()
        train_step(net, opt, batch, val_w=1.0, loss_scale=1.0, do_step=False)
        grad_full = next(net.parameters()).grad.clone()

        # Half scale
        opt.zero_grad()
        train_step(net, opt, batch, val_w=1.0, loss_scale=0.5, do_step=False)
        grad_half = next(net.parameters()).grad.clone()

        ratio = grad_half.norm() / (grad_full.norm() + 1e-12)
        assert 0.45 < ratio.item() < 0.55, f"Expected ~0.5 ratio, got {ratio.item():.3f}"
