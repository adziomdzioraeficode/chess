#!/usr/bin/env python3
"""
Compare 3 training stages: speed + quality smoke test.

Checkpoints:
  A) Run 1 final (pre-Phase 1)  = models/backup_run_20260420/mini_az_final.pt
  B) Run 2 final (post 5.1-5.5) = mini_az.pt.iter160.bak
  C) Current weights (= B, same weights, new code 5.9-5.11)

Tests:
  1. Loss evaluation on fixed data (forward-only) — measures model quality
  2. Training speed: 3 pipeline configs
     - "Pre-5.3":  batch=512, steps=200, accum=1 (old config)
     - "Post-5.3": batch=1024, steps=80, accum=1
     - "Current":  batch=1024, steps=40, accum=2, decisive=0.15
  3. Mini-training (20 steps) — measures gradient signal quality
     (how fast does loss drop with each config)
"""

import os
import sys
import time
import copy

import numpy as np
import torch
import chess

from mini_az.network import ChessNet
from mini_az.encoding import board_to_tensor, legal_moves_canonical
from mini_az.training import Sample, ReplayBuffer, collate, train_step


# ──────────────── helpers ────────────────

def make_diverse_buffer(n=3000):
    """Create a buffer with diverse positions (various openings + random z)."""
    buf = ReplayBuffer(100000)

    openings = [
        [],  # start pos
        ["e2e4"],
        ["d2d4"],
        ["e2e4", "e7e5"],
        ["d2d4", "d7d5"],
        ["e2e4", "c7c5"],
        ["g1f3", "d7d5"],
        ["e2e4", "e7e5", "g1f3", "b8c6"],
        ["d2d4", "g8f6", "c2c4", "e7e6"],
        ["e2e4", "e7e5", "f1c4"],
    ]

    for i in range(n):
        opening = openings[i % len(openings)]
        b = chess.Board()
        history = []
        for uci in opening:
            b.push_uci(uci)
            history.append(b.copy())

        t = board_to_tensor(b, history=history[-8:] if history else [])
        legals = legal_moves_canonical(b)
        if not legals:
            continue
        fs = np.array([m[0] for m in legals], dtype=np.int64)
        ts = np.array([m[1] for m in legals], dtype=np.int64)
        pr = np.array([m[2] for m in legals], dtype=np.int64)
        pi = np.ones(len(legals), dtype=np.float32) / len(legals)

        # Mix of decisive and drawn
        z_choices = [-1.0, -0.5, 0.0, 0.0, 0.5, 1.0]
        z = float(z_choices[i % len(z_choices)])
        if z > 0:
            wdl = np.array([0.8, 0.15, 0.05], dtype=np.float32)
        elif z < 0:
            wdl = np.array([0.05, 0.15, 0.8], dtype=np.float32)
        else:
            wdl = np.array([0.1, 0.8, 0.1], dtype=np.float32)

        plies_left = float(30 + (i % 60))
        buf.add_game([Sample(t.numpy(), fs, ts, pr, pi, z, wdl, plies_left)])

    return buf


def eval_loss(net, buf, batch_size=1024, n_batches=10):
    """Evaluate forward-only loss on fixed data (no grad)."""
    net.eval()
    losses = {"total": [], "pol": [], "val": [], "vz_corr": [], "pred_ent": []}

    with torch.no_grad():
        for _ in range(n_batches):
            batch_s = buf.sample_batch(batch_size)
            batch = collate(batch_s, "cpu")
            boards, fs, ts, pr, mask, target_pi, z, wdl_target, plies_left_target = batch

            logits, wdl_logits, moves_left_pred = net.forward_policy_value(boards, fs, ts, pr, mask)

            logp = torch.nn.functional.log_softmax(logits, dim=-1)
            pol_loss = -(target_pi * logp).masked_fill(~mask, 0.0).sum(dim=-1).mean()

            wdl_logp = torch.nn.functional.log_softmax(wdl_logits, dim=-1)
            val_loss = -(wdl_target * wdl_logp).sum(dim=-1).mean()

            pred_pi = torch.softmax(logits, dim=-1)
            log_pred = torch.log(pred_pi.clamp(min=1e-8))
            ent = -(pred_pi * log_pred).masked_fill(~mask, 0.0).sum(dim=-1).mean()

            wdl_probs = torch.softmax(wdl_logits, dim=-1)
            v = wdl_probs[:, 0] - wdl_probs[:, 2]
            vv = v.float()
            zz = z.float()
            if vv.std(unbiased=False) > 1e-6 and zz.std(unbiased=False) > 1e-6:
                vz = float(torch.corrcoef(torch.stack([vv, zz]))[0, 1].item())
            else:
                vz = 0.0

            losses["total"].append(float(pol_loss + 2.5 * val_loss))
            losses["pol"].append(float(pol_loss))
            losses["val"].append(float(val_loss))
            losses["vz_corr"].append(vz)
            losses["pred_ent"].append(float(ent))

    return {k: np.mean(v) for k, v in losses.items()}


def bench_train_speed(buf, batch_size, accum, steps, decisive_frac=0.0, label=""):
    """Measure training wall time for a given config (fresh net each time)."""
    torch.set_num_threads(24)
    net = ChessNet()
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

    # Warmup
    for _ in range(3):
        b = collate(buf.sample_batch(batch_size), "cpu")
        train_step(net, opt, b)

    loss_scale = 1.0 / accum
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        for _ in range(accum):
            if decisive_frac > 0:
                batch_s = buf.sample_batch_mixed(
                    batch_size, recent_frac=0.7, recent_window=2000,
                    decisive_frac=decisive_frac
                )
            else:
                batch_s = buf.sample_batch(batch_size)
            b = collate(batch_s, "cpu")
            train_step(net, opt, b, loss_scale=loss_scale, do_step=False)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
    elapsed = time.perf_counter() - t0
    ms_step = elapsed / steps * 1000
    return ms_step, elapsed


def mini_train(net, buf, batch_size, accum, steps, decisive_frac=0.0):
    """Quick training run, return per-step metrics to measure learning speed."""
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_scale = 1.0 / accum
    metrics_log = []

    for step in range(steps):
        opt.zero_grad()
        m = None
        for _ in range(accum):
            if decisive_frac > 0:
                batch_s = buf.sample_batch_mixed(
                    batch_size, recent_frac=0.7, recent_window=2000,
                    decisive_frac=decisive_frac
                )
            else:
                batch_s = buf.sample_batch(batch_size)
            b = collate(batch_s, "cpu")
            m = train_step(net, opt, b, val_w=2.5, loss_scale=loss_scale, do_step=False)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
        if m:
            metrics_log.append(m)

    return metrics_log


# ──────────────── main ────────────────

def main():
    torch.set_num_threads(24)
    print(f"CPU: {os.cpu_count()} cores | PyTorch: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")
    print()

    # Load checkpoints
    ckpts = {}
    paths = {
        "A: Run1 final (pre-Phase1)": "models/backup_run_20260420/mini_az_final.pt",
        "B: Run2 final (post 5.1-5.5)": "mini_az.pt.iter160.bak",
    }
    for label, path in paths.items():
        if os.path.exists(path):
            net = ChessNet()
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            ckpts[label] = net
            print(f"  ✓ Loaded {label}: {path}")
        else:
            print(f"  ✗ Missing {label}: {path}")

    if not ckpts:
        print("No checkpoints found!")
        return

    # Make reproducible test data
    torch.manual_seed(42)
    np.random.seed(42)
    buf = make_diverse_buffer(3000)
    print(f"\nTest buffer: {len(buf)} samples\n")

    # ═══════════════════════════════════════════════════════
    # 1. MODEL QUALITY: forward-only loss comparison
    # ═══════════════════════════════════════════════════════
    print("=" * 65)
    print("1. MODEL QUALITY (forward-only loss on same test data)")
    print("=" * 65)
    print(f"  {'Checkpoint':<35} {'pol_loss':>9} {'val_loss':>9} {'total':>8} {'vz_corr':>8} {'entropy':>8}")
    print("  " + "-" * 63)

    for label, net in ckpts.items():
        l = eval_loss(net, buf)
        print(f"  {label:<35} {l['pol']:>9.4f} {l['val']:>9.4f} {l['total']:>8.4f} "
              f"{l['vz_corr']:>8.4f} {l['pred_ent']:>8.3f}")
    print()

    # ═══════════════════════════════════════════════════════
    # 2. TRAINING SPEED: 3 pipeline configs
    # ═══════════════════════════════════════════════════════
    print("=" * 65)
    print("2. TRAINING SPEED (fresh net, 24 threads, no contention)")
    print("=" * 65)

    configs = [
        ("Pre-5.3:  batch=512, steps=200, accum=1", 512, 1, 20, 0.0),
        ("Post-5.3: batch=1024, steps=80, accum=1", 1024, 1, 20, 0.0),
        ("Current:  batch=1024, accum=2, dec=0.15", 1024, 2, 20, 0.15),
    ]

    print(f"  {'Config':<45} {'ms/step':>8} {'Proj iter':>10}")
    print("  " + "-" * 58)

    for label, bs, acc, bench_steps, dec in configs:
        ms, _ = bench_train_speed(buf, bs, acc, bench_steps, dec, label)
        # Project to real iter step counts
        if "Pre-5.3" in label:
            real_steps = 200
        elif "Post-5.3" in label:
            real_steps = 80
        else:
            real_steps = 40
        proj = ms * real_steps / 1000
        print(f"  {label:<45} {ms:>7.0f} {proj:>8.1f}s")
    print()

    # ═══════════════════════════════════════════════════════
    # 3. GRADIENT SIGNAL QUALITY: mini-train 20 steps
    # ═══════════════════════════════════════════════════════
    print("=" * 65)
    print("3. GRADIENT SIGNAL (20-step mini-train from Run2 weights)")
    print("=" * 65)

    best_ckpt_label = "B: Run2 final (post 5.1-5.5)"
    if best_ckpt_label not in ckpts:
        best_ckpt_label = list(ckpts.keys())[-1]
    base_net = ckpts[best_ckpt_label]

    train_configs = [
        ("Old: batch=1024, accum=1, dec=0.0", 1024, 1, 0.0),
        ("New: batch=1024, accum=2, dec=0.15", 1024, 2, 0.15),
    ]

    print(f"\n  Starting from: {best_ckpt_label}")
    print(f"  {'Config':<45} {'loss_0':>7} {'loss_19':>7} {'Δloss':>7} {'pol_0':>7} "
          f"{'pol_19':>7} {'Δpol':>7} {'vz_19':>6}")
    print("  " + "-" * 80)

    for label, bs, acc, dec in train_configs:
        net_copy = ChessNet()
        net_copy.load_state_dict(copy.deepcopy(base_net.state_dict()))
        torch.set_num_threads(24)

        metrics = mini_train(net_copy, buf, bs, acc, 20, dec)
        if len(metrics) >= 2:
            l0 = metrics[0]
            l19 = metrics[-1]
            dl = l19["loss"] - l0["loss"]
            dp = l19["pol"] - l0["pol"]
            print(f"  {label:<45} {l0['loss']:>7.4f} {l19['loss']:>7.4f} {dl:>+7.4f} "
                  f"{l0['pol']:>7.4f} {l19['pol']:>7.4f} {dp:>+7.4f} {l19['vz_corr']:>6.3f}")
    print()

    # ═══════════════════════════════════════════════════════
    # 4. DECISIVE MINING IMPACT: gradient magnitude
    # ═══════════════════════════════════════════════════════
    print("=" * 65)
    print("4. DECISIVE MINING: gradient magnitude comparison")
    print("=" * 65)

    net_dec = ChessNet()
    net_dec.load_state_dict(copy.deepcopy(base_net.state_dict()))
    net_dec.train()
    opt_dec = torch.optim.AdamW(net_dec.parameters(), lr=2e-4, weight_decay=1e-4)

    # Normal batch
    opt_dec.zero_grad()
    b_normal = collate(buf.sample_batch(1024), "cpu")
    m_normal = train_step(net_dec, opt_dec, b_normal, val_w=2.5, do_step=False)
    gnorm_normal = float(torch.nn.utils.clip_grad_norm_(net_dec.parameters(), float('inf')))
    opt_dec.zero_grad()

    # Decisive batch
    b_dec = collate(buf.sample_batch_mixed(1024, 0.7, 2000, decisive_frac=0.30), "cpu")
    m_dec = train_step(net_dec, opt_dec, b_dec, val_w=2.5, do_step=False)
    gnorm_dec = float(torch.nn.utils.clip_grad_norm_(net_dec.parameters(), float('inf')))

    print(f"  Normal batch:   grad_norm={gnorm_normal:.4f}  loss={m_normal['loss']:.4f}  pol={m_normal['pol']:.4f}")
    print(f"  Decisive batch: grad_norm={gnorm_dec:.4f}  loss={m_dec['loss']:.4f}  pol={m_dec['pol']:.4f}")
    print(f"  Grad magnitude boost: {gnorm_dec / gnorm_normal:.2f}x")
    print()

    # ═══════════════════════════════════════════════════════
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print("  • Model quality: compare pol_loss + vz_corr between Run1 and Run2")
    print("  • Speed: Current ~= Post-5.3 (grad_accum doesn't speed up, but")
    print("    gives 2x effective batch for gradient stability)")
    print("  • Gradient signal: decisive mining should show faster loss drop")
    print("  • Decisive mining: higher grad magnitude = stronger signal per step")


if __name__ == "__main__":
    main()
