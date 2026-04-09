#!/bin/bash
# tmux_run_learning.sh — Launch train_distill_then_rl.sh in a tmux session
# with full logging to file. Progress visible both in tmux and in .log file.
#
# Usage:
#   bash tmux_run_learning.sh              # default log: training_YYYYMMDD_HHMMSS.log
#   bash tmux_run_learning.sh my_run.log   # custom log file
#
set -euo pipefail

SESSION="learning"
LOGFILE="${1:-training_$(date +%Y%m%d_%H%M%S).log}"
SCRIPT="train_distill_then_rl.sh"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found in $(pwd)"
    exit 1
fi

# Kill existing session if running
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Kill it first with: tmux kill-session -t $SESSION"
    exit 1
fi

echo "Starting training in tmux session '$SESSION'"
echo "Log file: $(pwd)/$LOGFILE"
echo ""
echo "Useful commands:"
echo "  tmux attach -t $SESSION        # watch live progress"
echo "  tail -f $LOGFILE               # follow log file"
echo "  tmux kill-session -t $SESSION  # stop training"

tmux new-session -d -s "$SESSION" \
    "bash $SCRIPT 2>&1 | tee -a $LOGFILE; echo '=== DONE (exit \$?) ===' | tee -a $LOGFILE; sleep 999999"
