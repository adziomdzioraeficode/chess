#!/bin/bash
# run_tests.sh — Run test suite with coverage before making code changes.
#
# Usage:
#   ./run_tests.sh          # Run all tests with coverage report
#   ./run_tests.sh -x       # Stop on first failure
#   ./run_tests.sh -k mcts  # Run only MCTS tests
#
# This should be run before any code changes to ensure the baseline is green.
# Reference: lc0 project uses similar pre-commit testing to guard against regressions.
#
set -euo pipefail

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== Running mini_az test suite with coverage ==="
python -m pytest tests/ \
    --cov=mini_az \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    -v \
    --tb=short \
    "$@"

echo ""
echo "Coverage HTML report: htmlcov/index.html"
echo "=== Tests complete ==="
