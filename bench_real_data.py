#!/usr/bin/env python3
"""
Benchmark on REAL replay buffer data from Run 2.

Compares:
  A) Run1 final (pre-Phase1)
  B) Run2 final (post 5.1-5.5)
  + pipeline configs: old vs current

Uses actual game data from run 2 replay buffer — meaningful signal.
"""

import os
import sys
import time
import copy

import numpy as np
import torch
import chess

from mini_az.network import ChessNet
from mini_az.training import ReplayBuffer, collate, train_step


def eval_loss(net, buf, batch_size=1024, n_batches=20, use_mixed=False, decisive_frac=0.0):
    """Evaluate forward-only loss on real data."""
    net.eval()
    losses = {"total": [], "pol": [], "val": [], "vz_corr": [], "pred_ent": []}

    with torch.no_grad():
        for _ in range(n_batches):
            if use_mixed:
                batch_s = buf.sample_batch_mixed(
                    batch_size, recent_frac=0.7, recent_window=200000,
                    decisive_frac=decisive_frac
                )
            else:
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

    return {k: float(np.mean(v)) for k, v in losses.items()}


def mini_train(net, buf, batch_size, accum, steps, decisive_frac=0.0):
    """Quick training, return per-step metrics."""
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
                    batch_size, recent_frac=0.7, recent_window=200000,
                    decisive_frac=decisive_frac
                )
            else:
                batch_s = buf.sample_batch(batch_size)
            b = collate(batch_s, "cpu")
            m = train_step(net, opt, b, val_w=2.5, loss_scale=loss_scale, do_step=False)
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0))
        opt.step()
        if m:
            m["grad_norm"] = gn
            metrics_log.append(m)

    return metrics_log


def analyze_buffer(buf):
    """Quick stats on the replay buffer."""
    n = len(buf)
    zs = [s.z for s in buf.data[:min(n, 10000)]]
    decisive = sum(1 for z in zs if abs(z) > 0.01)
    draws = sum(1 for z in zs if abs(z) <= 0.01)
    avg_plies = np.mean([s.plies_left for s in buf.data[:min(n, 10000)]])
    avg_moves = np.mean([len(s.moves_fs) for s in buf.data[:min(n, 10000)]])
    return {
        "size": n,
        "decisive_pct": decisive / len(zs) * 100,
        "draw_pct": draws / len(zs) * 100,
        "avg_plies_left": avg_plies,
        "avg_legal_moves": avg_moves,
    }


def main():
    torch.set_num_threads(24)
    print(f"CPU: {os.cpu_count()} cores | Threads: {torch.get_num_threads()}")
    print()

    # ── Load replay buffer ──
    print("Loading replay buffer...")
    buf = ReplayBuffer.load("replay.pkl.gz")
    stats = analyze_buffer(buf)
    print(f"  Size: {stats['size']:,} samples")
    print(f"  Decisive: {stats['decisive_pct']:.1f}%  Draws: {stats['draw_pct']:.1f}%")
    print(f"  Avg legal moves: {stats['avg_legal_moves']:.1f}")
    print(f"  Avg plies left: {stats['avg_plies_left']:.1f}")
    print()

    # ── Load checkpoints ──
    ckpts = {}
    paths = {
        "A: Run1 (pre-Phase1)": "models/backup_run_20260420/mini_az_final.pt",
        "B: Run2 (post 5.1-5.5)": "mini_az.pt.iter160.bak",
    }
    for label, path in paths.items():
        if os.path.exists(path):
            net = ChessNet()
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            ckpts[label] = net
            print(f"  ✓ {label}")
        else:
            print(f"  ✗ Missing: {path}")

    if not ckpts:
        print("No checkpoints!")
        return
    print()

    # ═══════════════════════════════════════════════════════
    # 1. MODEL QUALITY on real data
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("1. MODEL QUALITY on real replay buffer data (20 batches × 1024)")
    print("=" * 70)
    print(f"  {'Model':<30} {'pol':>8} {'val':>8} {'total':>8} {'vz_corr':>8} {'entropy':>8}")
    print("  " + "-" * 64)

    for label, net in ckpts.items():
        l = eval_loss(net, buf, batch_size=1024, n_batches=20)
        print(f"  {label:<30} {l['pol']:>8.4f} {l['val']:>8.4f} {l['total']:>8.4f} "
              f"{l['vz_corr']:>8.4f} {l['pred_ent']:>8.3f}")
    print()

    # ═══════════════════════════════════════════════════════
    # 2. DECISIVE vs NORMAL sampling quality
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("2. DECISIVE MINING effect on loss (Run2 model)")
    print("=" * 70)

    best_label = "B: Run2 (post 5.1-5.5)"
    if best_label not in ckpts:
        best_label = list(ckpts.keys())[-1]
    net_b = ckpts[best_label]

    l_normal = eval_loss(net_b, buf, batch_size=1024, n_batches=20,
                         use_mixed=True, decisive_frac=0.0)
    l_dec15 = eval_loss(net_b, buf, batch_size=1024, n_batches=20,
                        use_mixed=True, decisive_frac=0.15)
    l_dec30 = eval_loss(net_b, buf, batch_size=1024, n_batches=20,
                        use_mixed=True, decisive_frac=0.30)

    print(f"  {'Sampling':<30} {'pol':>8} {'val':>8} {'total':>8} {'vz_corr':>8}")
    print("  " + "-" * 50)
    print(f"  {'Normal (dec=0.0)':<30} {l_normal['pol']:>8.4f} {l_normal['val']:>8.4f} "
          f"{l_normal['total']:>8.4f} {l_normal['vz_corr']:>8.4f}")
    print(f"  {'Decisive (dec=0.15)':<30} {l_dec15['pol']:>8.4f} {l_dec15['val']:>8.4f} "
          f"{l_dec15['total']:>8.4f} {l_dec15['vz_corr']:>8.4f}")
    print(f"  {'Decisive (dec=0.30)':<30} {l_dec30['pol']:>8.4f} {l_dec30['val']:>8.4f} "
          f"{l_dec30['total']:>8.4f} {l_dec30['vz_corr']:>8.4f}")
    print()

    # ═══════════════════════════════════════════════════════
    # 3. MINI-TRAIN: 30 steps, compare configs
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("3. MINI-TRAIN (30 steps from Run2 weights, real data)")
    print("=" * 70)

    train_configs = [
        ("Old: b=1024, acc=1, dec=0.0", 1024, 1, 0.0),
        ("New: b=1024, acc=2, dec=0.0", 1024, 2, 0.0),
        ("New: b=1024, acc=2, dec=0.15", 1024, 2, 0.15),
    ]

    STEPS = 30
    print(f"\n  {STEPS} steps from {best_label}")
    print(f"  {'Config':<35} {'loss_0':>8} {'loss_end':>8} {'Δloss':>8} "
          f"{'pol_0':>7} {'pol_end':>7} {'Δpol':>7} {'vz_end':>7} {'gnorm':>7}")
    print("  " + "-" * 90)

    for label, bs, acc, dec in train_configs:
        net_copy = ChessNet()
        net_copy.load_state_dict(copy.deepcopy(net_b.state_dict()))
        torch.set_num_threads(24)

        t0 = time.perf_counter()
        metrics = mini_train(net_copy, buf, bs, acc, STEPS, dec)
        elapsed = time.perf_counter() - t0

        if len(metrics) >= 2:
            m0 = metrics[0]
            mf = metrics[-1]
            dl = mf["loss"] - m0["loss"]
            dp = mf["pol"] - m0["pol"]
            avg_gn = np.mean([m["grad_norm"] for m in metrics])
            print(f"  {label:<35} {m0['loss']:>8.4f} {mf['loss']:>8.4f} {dl:>+8.4f} "
                  f"{m0['pol']:>7.4f} {mf['pol']:>7.4f} {dp:>+7.4f} {mf['vz_corr']:>7.3f} {avg_gn:>7.3f}"
                  f"  ({elapsed:.1f}s)")

    # ── Also check post-train eval quality ──
    print(f"\n  Post-train eval (batch=1024, 10 batches):")
    print(f"  {'Config':<35} {'pol':>8} {'val':>8} {'total':>8} {'vz_corr':>8}")
    print("  " + "-" * 55)

    for label, bs, acc, dec in train_configs:
        net_copy = ChessNet()
        net_copy.load_state_dict(copy.deepcopy(net_b.state_dict()))
        torch.set_num_threads(24)
        mini_train(net_copy, buf, bs, acc, STEPS, dec)
        l = eval_loss(net_copy, buf, batch_size=1024, n_batches=10)
        print(f"  {label:<35} {l['pol']:>8.4f} {l['val']:>8.4f} {l['total']:>8.4f} {l['vz_corr']:>8.4f}")

    print()

    # ═══════════════════════════════════════════════════════
    # 4. SPEED
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("4. TRAINING SPEED (24 threads, no contention)")
    print("=" * 70)

    speed_configs = [
        ("Pre-5.3:  b=512, steps=200", 512, 1, 200, 0.0),
        ("Post-5.3: b=1024, steps=80", 1024, 1, 80, 0.0),
        ("Current:  b=1024, acc=2, s=40", 1024, 2, 40, 0.15),
    ]

    print(f"  {'Config':<35} {'ms/step':>9} {'Proj iter':>11} {'eff_batch':>10}")
    print("  " + "-" * 60)

    for label, bs, acc, real_steps, dec in speed_configs:
        net_tmp = ChessNet()
        net_tmp.train()
        opt_tmp = torch.optim.AdamW(net_tmp.parameters(), lr=2e-4, weight_decay=1e-4)
        loss_scale = 1.0 / acc

        # Warmup
        for _ in range(3):
            b = collate(buf.sample_batch(bs), "cpu")
            train_step(net_tmp, opt_tmp, b)

        bench_steps = min(15, real_steps)
        t0 = time.perf_counter()
        for _ in range(bench_steps):
            opt_tmp.zero_grad()
            for _ in range(acc):
                if dec > 0:
                    batch_s = buf.sample_batch_mixed(bs, 0.7, 200000, decisive_frac=dec)
                else:
                    batch_s = buf.sample_batch(bs)
                b = collate(batch_s, "cpu")
                train_step(net_tmp, opt_tmp, b, loss_scale=loss_scale, do_step=False)
            torch.nn.utils.clip_grad_norm_(net_tmp.parameters(), 3.0)
            opt_tmp.step()
        elapsed = time.perf_counter() - t0
        ms = elapsed / bench_steps * 1000
        proj = ms * real_steps / 1000
        print(f"  {label:<35} {ms:>8.0f} {proj:>9.1f}s {bs * acc:>10}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
