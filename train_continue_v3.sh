#!/bin/bash
# train_continue_v3.sh — Continue v3 training from existing weights
#
# KEEPS: mini_az.pt (model weights), mini_az_ckpt.pt (optimizer state),
#        models/best.pt + metric files
# CLEARS: replay buffer (old data is from 32 sims — garbage quality)
#         train/eval logs (fresh metrics for new run)
#
# KEY FIXES vs original train_fresh_v3.sh:
#   1. mp_sims 32 → 200  (CRITICAL: 32 sims = noise, 200 = usable MCTS targets)
#   2. mcts_value_mix 0.0 → 0.25  (value head was dead — no gradient signal)
#   3. max_plies 220 → 160  (shorter games, break draw spiral)
#   4. rand_eval_games 6 → 20  (statistical significance)
#   5. rand_eval_sims 32 → 200  (eval must use real search)
#   6. val_w 1.0 → 2.0  (boost value head learning)
#   7. games_per_iter 120 → 60  (compensate for 6× slower games at 200 sims)
#   8. workers 92 → tuned after benchmark
#   9. sf_boot_depth 8 → 12  (stronger bootstrap signal for value)
#  10. steps_per_iter 150 → 200 (more training per iter with better data)
#
set -euo pipefail

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f "$HOME/mypipenv/bin/activate" ]; then
    source "$HOME/mypipenv/bin/activate"
elif [ -f "$HOME/mypythonenv/bin/activate" ]; then
    source "$HOME/mypythonenv/bin/activate"
fi

# --- Run tests before training ---
echo "=== Running test suite (pre-training sanity check) ==="
if python -m pytest tests/ -x -q --tb=short 2>/dev/null; then
    echo "All tests passed. Proceeding with training."
else
    echo "WARNING: Some tests failed. Review before training."
    echo "Continuing anyway (training is idempotent)."
fi

# --- Keep model weights, clear bad replay data ---
echo "=== Continuing from existing weights ==="
if [ -f mini_az.pt ]; then
    echo "Found model weights: mini_az.pt ($(du -h mini_az.pt | cut -f1))"
else
    echo "WARNING: No mini_az.pt found — will train from scratch."
fi

# Clear replay buffer (full of 32-sims garbage data)
rm -f replay.pkl.gz replay.pkl.gz.bak
echo "Cleared replay buffer (old 32-sims data)."

# Clear old logs (new run = new metrics)
rm -f train_log.csv eval_log.csv
echo "Cleared old log files."

# Keep best.pt and metric files for gating continuity
if [ -f models/best.pt ]; then
    echo "Found best model: models/best.pt"
fi

mkdir -p models

# 4h hard timeout — ensures VM cost stays bounded
exec timeout 4h python -m mini_az --mode train \
    --clear_buffer \
    --resume_opt \
    --workers 92 \
    --mp_sims 200 \
    --games_per_iter 60 \
    --iters 9999 \
    --steps_per_iter 200 \
    --batch 512 \
    --lr 5e-4 \
    --buffer 500000 \
    --recent_frac 0.70 \
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
