#!/usr/bin/env bash
# ===========================================================================
# train.sh  -  Launch GRPO training with VeRL
#
# Usage:
#   # Smoke test (minimal config, fast verification)
#   bash experiments/grpo_ale_generalization/train.sh --smoke
#
#   # Main training
#   bash experiments/grpo_ale_generalization/train.sh
#
#   # Custom VeRL config
#   bash experiments/grpo_ale_generalization/train.sh --verl_config path/to/verl.yaml
# ===========================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
EXP_CONFIG="${SCRIPT_DIR}/configs/exp.yaml"
VERL_CONFIG="${SCRIPT_DIR}/configs/verl_grpo.yaml"
SMOKE=false
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=true
            shift
            ;;
        --verl_config)
            VERL_CONFIG="$2"
            shift 2
            ;;
        --config)
            EXP_CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export ALE_EXP_CONFIG="$EXP_CONFIG"

echo "=== GRPO ALE-Bench Training ==="
echo "  Repo root:    $REPO_ROOT"
echo "  Exp config:   $EXP_CONFIG"
echo "  VeRL config:  $VERL_CONFIG"
echo "  Smoke mode:   $SMOKE"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Verify remote evaluator is accessible
# ---------------------------------------------------------------------------
EVAL_URL=$(python3 -c "
import yaml
with open('$EXP_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('remote_evaluator_url', 'http://10.214.54.87:8000'))
")

echo "[1/3] Checking remote evaluator at $EVAL_URL ..."
HEALTH=$(curl -s --max-time 5 "$EVAL_URL/health" 2>/dev/null || echo '{"status":"unreachable"}')
echo "  Health: $HEALTH"
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok'" 2>/dev/null; then
    echo "  Evaluator OK"
else
    echo "  WARNING: Remote evaluator not reachable at $EVAL_URL"
    echo "  Training will fail on reward computation."
    echo "  Start the evaluator first: bash experiments/grpo_ale_generalization/remote_evaluator/start_devhost.sh"
    echo ""
    read -p "  Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Generate dataset if not present
# ---------------------------------------------------------------------------
DATASET_PATH="${SCRIPT_DIR}/data/train_data/dataset.jsonl"
echo ""
echo "[2/3] Checking dataset..."
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "  Dataset not found. Generating..."
    python3 "${SCRIPT_DIR}/data/make_dataset.py" --config "$EXP_CONFIG" --no_evaluator
else
    echo "  Dataset exists: $DATASET_PATH"
    echo "  (To regenerate: rm $DATASET_PATH && rerun)"
fi

# ---------------------------------------------------------------------------
# Step 3: Launch VeRL GRPO training
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Launching training..."

# Build the training command
# VeRL typically uses a Python entry point. We construct the command
# based on whether VeRL is installed and the training mode.

# Check if verl is available
if python3 -c "import verl" 2>/dev/null; then
    echo "  VeRL detected."
    TRAIN_CMD="python3 -m verl.trainer.main_grpo"
else
    echo "  VeRL not detected. Using standalone training script."
    TRAIN_CMD="python3 ${SCRIPT_DIR}/train_standalone.py"
fi

if [[ "$SMOKE" == "true" ]]; then
    echo "  Mode: SMOKE TEST (minimal steps)"
    # Override key parameters for smoke test
    SMOKE_OVERRIDES="
        --trainer.total_steps=4
        --trainer.batch_size=2
        --trainer.grpo.group_size=2
        --trainer.gradient_accumulation_steps=1
    "
    # Run with smoke overrides
    $TRAIN_CMD \
        --config "$VERL_CONFIG" \
        $SMOKE_OVERRIDES \
        $EXTRA_ARGS \
        2>&1 | tee "${SCRIPT_DIR}/logs/train_smoke_$(date +%Y%m%d_%H%M%S).log"
else
    echo "  Mode: MAIN TRAINING"
    $TRAIN_CMD \
        --config "$VERL_CONFIG" \
        $EXTRA_ARGS \
        2>&1 | tee "${SCRIPT_DIR}/logs/train_main_$(date +%Y%m%d_%H%M%S).log"
fi

echo ""
echo "=== Training complete ==="
