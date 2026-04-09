#!/bin/bash
# train_distill_then_rl.sh — Two-phase training:
#   Phase 1: Supervised distillation from Stockfish (fast, ~30min)
#   Phase 2: RL self-play refinement with MCTS (remaining time)
#
# This is the "NNUE approach" — first learn position evaluation from a strong
# teacher (Stockfish), then refine with self-play. ~100x faster bootstrap
# than pure RL from random initialization.
#
set -euo pipefail

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f "$HOME/mypipenv/bin/activate" ]; then
    source "$HOME/mypipenv/bin/activate"
elif [ -f "$HOME/mypythonenv/bin/activate" ]; then
    source "$HOME/mypythonenv/bin/activate"
fi

echo "=== Running test suite ==="
if python -m pytest tests/ -x -q --tb=short 2>/dev/null; then
    echo "All tests passed."
else
    echo "WARNING: Some tests failed."
fi

export PYTHONUNBUFFERED=1
mkdir -p models

# -------------------------------------------------------
# PHASE 1: Supervised distillation from Stockfish (~30 min)
# -------------------------------------------------------
# Generate 300K positions with SF depth-12 eval + depth-10 multipv policy
# Train 3 epochs supervised. This bootstraps policy + value head to SF level.
# On 96 vCPU: ~10K samples/min = 300K in ~30 min
echo ""
echo "=========================================="
echo "PHASE 1: Supervised Distillation from SF"
echo "=========================================="

timeout 45m python -u -m mini_az --mode distill \
    --workers 90 \
    --distill_samples 300000 \
    --distill_epochs 3 \
    --distill_eval_depth 12 \
    --distill_policy_depth 10 \
    --distill_multipv 8 \
    --distill_positions_per_game 10 \
    --distill_max_random_plies 120 \
    --batch 512 \
    --lr 1e-3 \
    --sf_path /usr/games/stockfish \
    --sf_elo 2000 \
    --sf_cp_scale 600.0 \
    --sf_cp_cap 1200 \
    --val_w 2.0 \
    --moves_left_w 0.15 \
    "$@"

echo ""
echo "Distillation complete. Model saved to mini_az.pt"
echo ""

# Copy distilled model as best.pt for RL gating baseline
cp -f mini_az.pt models/best.pt
rm -f models/best.pt.metric models/best.pt.metric_kind
echo "Set distilled model as best.pt baseline."

# -------------------------------------------------------
# PHASE 2: RL self-play refinement (remaining ~3h)
# -------------------------------------------------------
# Now the network has SF-quality priors. MCTS search will be meaningful
# because the policy is good and value is calibrated. RL refines beyond
# what pure imitation can achieve (tactics, sacrifices, long-term plans).
echo ""
echo "=========================================="
echo "PHASE 2: RL Self-Play Refinement"
echo "=========================================="

# Clear old RL data
rm -f replay.pkl.gz replay.pkl.gz.bak
rm -f train_log.csv eval_log.csv

exec timeout 3h python -u -m mini_az --mode train \
    --clear_buffer \
    --workers 92 \
    --mp_sims 128 \
    --games_per_iter 80 \
    --iters 9999 \
    --steps_per_iter 200 \
    --batch 512 \
    --lr 3e-4 \
    --buffer 500000 \
    --recent_frac 0.70 \
    --recent_window 200000 \
    --max_plies 160 \
    --resign_threshold -0.95 \
    --resign_patience 8 \
    --sf_path /usr/games/stockfish \
    --sf_elo 2000 \
    --sf_eval_elo 1320 \
    --sf_eval_elo_easy 0 \
    --sf_eval_max_plies 200 \
    --sf_worker_frac 0.70 \
    --sf_boot_prob 1.0 \
    --sf_boot_time_ms 0 \
    --sf_boot_depth 12 \
    --sf_cp_scale 600.0 \
    --sf_cp_cap 1000 \
    --sf_teacher_prob 0.60 \
    --sf_teacher_mix 0.35 \
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
