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

# ── 4. Pull LFS files ──
echo "=== 4/5: Pulling Git LFS files ==="
git lfs pull
echo "  ✓ LFS pull complete"
echo ""

# ── 5. Check training artifacts ──
echo "=== 5/5: Checking training artifacts ==="
echo ""

ARTIFACTS=(
    "mini_az.pt:model weights"
    "mini_az_ckpt.pt:optimizer checkpoint"
    "models/best.pt:best gated model"
    "replay.pkl.gz:replay buffer"
)

all_ok=true
for entry in "${ARTIFACTS[@]}"; do
    path="${entry%%:*}"
    desc="${entry#*:}"
    if [ -f "$path" ]; then
        echo "  ✓ $path ($(du -h "$path" | cut -f1)) — $desc"
    else
        echo "  ✗ $path MISSING — $desc"
        all_ok=false
    fi
done

if [ "$all_ok" = true ]; then
    echo ""
    echo "  All artifacts present — full resume possible."
else
    echo ""
    echo "  Some artifacts missing. Training will still work but may"
    echo "  reset optimizer state or start with empty replay buffer."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start training:"
echo "  bash tmux_run_learning.sh"
echo ""
echo "To monitor:"
echo "  tmux attach -t learning"
echo "  tail -f training_*.log"
echo ""
