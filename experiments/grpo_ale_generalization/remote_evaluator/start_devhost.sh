#!/usr/bin/env bash
# ===========================================================================
# start_devhost.sh  -  Start the remote evaluator server
# Run on: 10.214.54.87
# ===========================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PORT="${ALE_EVAL_PORT:-8000}"
MAX_CONCURRENT="${ALE_EVAL_MAX_CONCURRENT:-4}"
CACHE_DIR="${ALE_EVAL_CACHE:-/data/ale_bench_cache}"
API_KEY="${ALE_EVAL_API_KEY:-}"
LOG_FILE="${SCRIPT_DIR}/evaluator.log"

export ALE_EVAL_PORT="$PORT"
export ALE_EVAL_MAX_CONCURRENT="$MAX_CONCURRENT"
export ALE_EVAL_CACHE="$CACHE_DIR"
export ALE_EVAL_API_KEY="$API_KEY"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "=== Starting ALE-Bench Remote Evaluator ==="
echo "  Port:           $PORT"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Cache dir:      $CACHE_DIR"
echo "  Log file:       $LOG_FILE"
echo ""

# Option 1: Run in foreground (Ctrl+C to stop)
# python3 "$SCRIPT_DIR/evaluator_server.py"

# Option 2: Run in background with nohup
echo "Starting in background (nohup)..."
nohup python3 "$SCRIPT_DIR/evaluator_server.py" > "$LOG_FILE" 2>&1 &
PID=$!
echo "  PID: $PID"
echo "  To stop: kill $PID"
echo "  To monitor: tail -f $LOG_FILE"
echo ""

# Wait a moment and check
sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "Server started successfully."
    echo "Health check: curl http://localhost:$PORT/health"
else
    echo "ERROR: Server failed to start. Check $LOG_FILE"
    tail -20 "$LOG_FILE"
    exit 1
fi
