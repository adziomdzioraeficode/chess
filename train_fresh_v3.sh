#!/bin/bash
# train_fresh_v3.sh — v3: 45 planes + 10×96 SE-ResNet (~5.5M) + WDL + dynamic c_puct
#
# V3 ARCHITECTURE:
#   - 45 input planes: 3 positions × 12 piece planes + 9 aux (castling, turn, halfmove, fullmove, check, repetition)
#   - 10 SE-residual blocks × 96 channels (~5.5M parameters)
#   - WDL value head (win/draw/loss softmax) — replaces single tanh
#   - Move-embedding policy head (from/to/promo embeddings)
#   - Dynamic c_puct: log((1+N+19652)/19652) + 2.5  (AlphaZero formula)
#   - Repetition plane: 0.0 (new) / 0.5 (2nd occurrence) / 1.0 (threefold)
#   - Halfmove clock scaled /50 (chess rule: 50-move draw)
#   - Clean pi_target: MCTS visits + SF teacher blend, no repetition penalty
#   - Behavior policy: keeps anti-repetition penalty for exploration
#   - mcts_value_mix=0.25: target = 75% game outcome + 25% MCTS value
#
# INCOMPATIBLE with v1/v2 models.  Clean start required.
#
# Target: Azure Standard_D96s_v6 (48 physical / 96 HT cores, AMD EPYC 9004)
#
# Tuning rationale (v3 vs v2):
#   - Network 2× bigger (5.5M vs 3.3M) → inference ~1.7× slower per search
#   - 80 workers optimal on D96s v6 (bench: 60 searches/s at 64 sims)
#   - Training uses 32 threads during pause (bench: 5.0 steps/s@batch512)
#   - SF teacher depth-14 multipv-5: ~148ms/call, 80% prob → ~40% effective blend
#   - Expected iter time: ~140s selfplay + ~20s train = ~160s/iter
#   - 8h ≈ ~180 iters
#
# Strategy:
#   - Phase 1 (iter 1-30):  Heavy SF distillation, random init → basic piece values
#   - Phase 2 (iter 30-80): Network starts learning tactics from SF blend
#   - Phase 3 (iter 80+):   Cosine LR decay, should start beating random consistently
#   - Target: beat SF Skill Level 0 (~1000 Elo) by iter ~100
#
set -euo pipefail

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Clean old incompatible artifacts (new architecture v3)
echo "=== Cleaning old artifacts for v3 architecture ==="
rm -f mini_az.pt mini_az_ckpt.pt replay.pkl.gz
rm -f models/best.pt models/best.pt.metric models/best.pt.metric_kind models/opponent.pt
echo "Done. Starting v3 training from scratch."

mkdir -p models

exec python -m mini_az --mode train \
    --workers 80 \
    --mp_sims 64 \
    --games_per_iter 120 \
    --iters 9999 \
    --steps_per_iter 100 \
    --batch 512 \
    --lr 1e-3 \
    --buffer 500000 \
    --recent_frac 0.75 \
    --recent_window 200000 \
    --max_plies 150 \
    --resign_threshold -0.95 \
    --resign_patience 10 \
    --sf_path /usr/games/stockfish \
    --sf_elo 2000 \
    --sf_eval_elo 1320 \
    --sf_eval_elo_easy 0 \
    --sf_boot_prob 1.0 \
    --sf_boot_time_ms 0 \
    --sf_boot_depth 8 \
    --sf_cp_scale 600.0 \
    --sf_cp_cap 1000 \
    --sf_teacher_prob 0.80 \
    --sf_teacher_mix 0.50 \
    --sf_teacher_time_ms 0 \
    --sf_teacher_depth 14 \
    --sf_teacher_multipv 5 \
    --sf_teacher_cp_cap 600 \
    --sf_teacher_cp_soft_scale 150.0 \
    --sf_teacher_eps 0.02 \
    --mcts_value_mix 0.25 \
    --mix_best 0.25 \
    --mix_opp 0.15 \
    --opp_lag 5 \
    --eval_every 5 \
    --eval_games 10 \
    --eval_sims 400 \
    --sf_movetime_ms 10 \
    --rand_eval_games 6 \
    --rand_eval_sims 32 \
    --rand_max_plies 120 \
    --self_eval_games 10 \
    --self_eval_sims 32 \
    --self_eval_max_plies 120 \
    --save_every 5 \
    --val_w 1.0 \
    --gate_margin 0.005 \
    "$@"
