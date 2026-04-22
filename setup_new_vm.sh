#!/bin/bash
# setup_new_vm.sh — Bootstrap a fresh VM for mini_az chess training.
#
# Run after git clone:
#   git clone <repo-url> chess && cd chess
#   bash setup_new_vm.sh
#
# What this does:
#   1. Install system deps (stockfish, tmux, python3-venv)
#   2. Create Python venv and install pip packages
#   3. Run tests to verify everything works
#   4. Warn about missing gitignored artifacts
#
# After setup, start training with:
#   bash tmux_run_learning.sh
#
set -euo pipefail

echo "============================================"
echo "  mini_az chess RL — New VM Setup"
echo "============================================"
echo ""

# ── 1. System packages ──
echo "=== 1/4: Installing system dependencies ==="
if ! command -v stockfish &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq stockfish tmux python3-venv
    echo "  ✓ Installed stockfish, tmux, python3-venv"
else
    echo "  ✓ stockfish already installed: $(which stockfish)"
fi

if ! command -v tmux &>/dev/null; then
    sudo apt-get install -y -qq tmux
fi
echo "  ✓ tmux: $(tmux -V)"
echo ""

# ── 2. Python environment ──
echo "=== 2/4: Setting up Python environment ==="
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "  ✓ venv already exists: $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Created venv: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "  Python: $(python --version) @ $(which python)"

pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  ✓ Installed pip packages"
echo ""

# Verify key packages
echo "  Installed versions:"
python -c "
import torch, chess, numpy, tqdm
print(f'    torch:  {torch.__version__}')
print(f'    chess:  {chess.__version__}')
print(f'    numpy:  {numpy.__version__}')
print(f'    tqdm:   {tqdm.__version__}')
print(f'    CUDA:   {torch.cuda.is_available()}')
print(f'    CPUs:   {torch.get_num_threads()} threads')
"
echo ""

# ── 3. Run tests ──
echo "=== 3/4: Running test suite ==="
if python -m pytest tests/ -x -q --tb=short; then
    echo "  ✓ All tests passed"
else
    echo "  ✗ Some tests failed — review before training!"
fi
echo ""

# ── 4. Check training artifacts ──
echo "=== 4/4: Checking training artifacts ==="
echo ""

# Model weights (TRACKED — should be there)
if [ -f mini_az.pt ]; then
    echo "  ✓ mini_az.pt ($(du -h mini_az.pt | cut -f1)) — model weights"
else
    echo "  ✗ mini_az.pt MISSING — training will start from scratch!"
fi

if [ -f models/best.pt ]; then
    echo "  ✓ models/best.pt ($(du -h models/best.pt | cut -f1)) — best gated model"
else
    echo "  ⚠ models/best.pt missing — will be created during training"
fi

# Checkpoint (GITIGNORED — won't be there)
if [ -f mini_az_ckpt.pt ]; then
    echo "  ✓ mini_az_ckpt.pt ($(du -h mini_az_ckpt.pt | cut -f1)) — optimizer checkpoint"
else
    echo "  ⚠ mini_az_ckpt.pt MISSING (gitignored)"
    echo "    → Training will work but optimizer state (Adam moments) resets."
    echo "    → To preserve optimizer: scp from old VM or remove --resume_opt"
    echo "    → Without it: first ~50 iters may have slightly noisier gradients."
    echo "    → This is NOT critical — AdamW will re-adapt in ~20-50 iters."
fi

# Replay buffer (GITIGNORED — won't be there)
if [ -f replay.pkl.gz ]; then
    echo "  ✓ replay.pkl.gz ($(du -h replay.pkl.gz | cut -f1)) — replay buffer"
else
    echo "  ⚠ replay.pkl.gz MISSING (gitignored)"
    echo "    → Training starts with empty buffer — first 2-3 iters will be slow"
    echo "    → This is fine: buffer fills from selfplay and doesn't need old data."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start training:"
echo "  bash tmux_run_learning.sh"
echo ""
echo "To copy optimizer state from old VM (optional):"
echo "  scp old-vm:~/repo/chess/mini_az_ckpt.pt ."
echo ""
echo "To monitor:"
echo "  tmux attach -t learning"
echo "  tail -f training_*.log"
echo ""
