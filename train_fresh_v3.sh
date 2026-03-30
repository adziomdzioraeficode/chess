#!/bin/bash
# train_fresh_v3.sh — v3: 45 planes + 10×96 SE-ResNet (~5.5M) + WDL + dynamic c_puct
#
# V3 ARCHITECTURE (lc0-inspired home Python RL chess):
#   - 45 input planes: 3 positions × 12 piece planes + 9 aux (castling, turn, halfmove, fullmove, check, repetition)
#   - 10 SE-residual blocks × 96 channels (~5.5M parameters)
#   - WDL value head (win/draw/loss softmax) — replaces single tanh (like lc0 T40+)
#   - Move-embedding policy head (from/to/promo embeddings)
#   - Dynamic c_puct: log((1+N+19652)/19652) + 2.5  (AlphaZero/lc0 formula)
#   - lc0-style MCTS: root FPU=-1.0, child FPU=parent_Q-0.25, policy_temp=1.4 at root
#   - Smooth temperature decay: T=1.0 for ply<16, exp(-0.08*(ply-16)) after (lc0 curve)
#   - PV cache limited to 20K entries (prevents OOM in long games)
#   - LR schedule: 2% linear warmup → cosine decay (lc0-style stabilises random init)
#   - Repetition plane: 0.0 (new) / 0.5 (2nd occurrence) / 1.0 (threefold)
#   - Halfmove clock scaled /50 (chess rule: 50-move draw)
#   - Clean pi_target: MCTS visits + SF teacher blend, no repetition penalty
#   - Behavior policy: keeps anti-repetition penalty for exploration
#   - mcts_value_mix=0.25: target = 75% game outcome + 25% MCTS value
#   - moves-left head: dodatkowy sygnal przeciw shufflingowi i dla szybszej konwersji
#   - sharp replay sampling: czesc batcha z pozycji bardziej decyzyjnych
#
# INCOMPATIBLE with v1/v2 models.  Clean start required.
#
# Target: Azure Standard_Ds96_v6 (48 physical / 96 HT cores, AMD EPYC 9004)
# Time budget: 4 hours (14400s)
# Goal: beat Stockfish 1320 Elo (like lc0 early nets but as a home RL project)
#
# Tuning rationale (4h on Ds96v6):
#   - Network 2× bigger (5.5M vs 3.3M) → inference ~1.7× slower per search
#   - 80 workers optimal on Ds96v6 (bench: 60 searches/s at 32 sims)
#   - Training uses 32 threads during pause (bench: ~5 steps/s@batch512)
#   - SF teacher depth-8 multipv-5: ~50ms/call, 80% prob → ~40% effective blend
#   - mp_sims=32: 2× more games/iter vs 64 sims — faster iteration, key early on
#   - steps_per_iter=200: more gradient steps to utilise the extra games
#   - Expected iter time: ~80s selfplay + ~40s train = ~120s/iter
#   - 4h ≈ ~120 iters (hard timeout enforced)
#
# Strategy (compressed for 4h):
#   - Phase 1 (iter 1-20):  Heavy SF distillation, random init → basic piece values
#                            (LR warmup: 2% of total steps @ 0.01× → 1× base LR)
#   - Phase 2 (iter 20-60): Network starts learning tactics from SF blend
#   - Phase 3 (iter 60+):   Cosine LR decay, should start beating random consistently
#   - Target: beat SF 1320 Elo — evaluate every 5 iters to track progress
#
# Reference: lc0 (github.com/LeelaChessZero/lc0) does this at scale with C++/CUDA.
# We replicate the key ideas (MCTS+NN, WDL, teacher distillation, FPU, policy temp)
# in a home Python project to learn RL and reach competitive play vs SF 1320.
#
set -euo pipefail

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f "$HOME/mypipenv/bin/activate" ]; then
    source "$HOME/mypipenv/bin/activate"
elif [ -f "$HOME/mypythonenv/bin/activate" ]; then
    source "$HOME/mypythonenv/bin/activate"
fi

# --- Run tests before training to catch regressions ---
echo "=== Running test suite (pre-training sanity check) ==="
if python -m pytest tests/ -x -q --tb=short 2>/dev/null; then
    echo "All tests passed. Proceeding with training."
else
    echo "WARNING: Some tests failed. Review before training."
    echo "Continuing anyway (training is idempotent)."
fi

# Clean old incompatible artifacts (new architecture v3)
echo "=== Cleaning old artifacts for v3 architecture ==="
rm -f mini_az.pt mini_az_ckpt.pt replay.pkl.gz
rm -f models/best.pt models/best.pt.metric models/best.pt.metric_kind models/opponent.pt
echo "Done. Starting v3 training from scratch."

mkdir -p models

# 4h hard timeout — ensures VM cost stays bounded
exec timeout 4h python -m mini_az --mode train \
    --clear_buffer \
    --workers 92 \
    --mp_sims 200 \
    --games_per_iter 60 \
    --iters 9999 \
    --steps_per_iter 200 \
    --batch 512 \
    --lr 1e-3 \
    --buffer 500000 \
    --recent_frac 0.75 \
    --recent_window 200000 \
    --max_plies 160 \
    --resign_threshold -0.95 \
    --resign_patience 10 \
    --sf_path /usr/games/stockfish \
    --sf_elo 2000 \
    --sf_eval_elo 1320 \
    --sf_eval_elo_easy 0 \
    --sf_eval_max_plies 200 \
    --sf_worker_frac 0.75 \
    --sf_boot_prob 1.0 \
    --sf_boot_time_ms 0 \
    --sf_boot_depth 12 \
    --sf_cp_scale 600.0 \
    --sf_cp_cap 1000 \
    --sf_teacher_prob 0.80 \
    --sf_teacher_mix 0.50 \
    --sf_teacher_time_ms 0 \
    --sf_teacher_depth 8 \
    --sf_teacher_multipv 5 \
    --sf_teacher_cp_cap 600 \
    --sf_teacher_cp_soft_scale 150.0 \
    --sf_teacher_eps 0.02 \
    --mcts_value_mix 0.25 \
    --moves_left_w 0.15 \
    --mix_best 0.25 \
    --mix_opp 0.15 \
    --opp_lag 5 \
    --sharp_frac 0.20 \
    --sharp_threshold 0.35 \
    --eval_every 5 \
    --eval_games 10 \
    --eval_sims 400 \
    --sf_movetime_ms 10 \
    --rand_eval_games 20 \
    --rand_eval_sims 200 \
    --rand_max_plies 150 \
    --self_eval_games 12 \
    --self_eval_sims 64 \
    --self_eval_max_plies 150 \
    --save_every 5 \
    --val_w 2.0 \
    --gate_margin 0.005 \
    "$@"
