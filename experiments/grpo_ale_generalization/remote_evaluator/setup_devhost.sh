#!/usr/bin/env bash
# ===========================================================================
# setup_devhost.sh  -  Prepare the development host for running the evaluator
# Run on: 10.214.54.87 (or any host with Docker)
# ===========================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== ALE-Bench Remote Evaluator Setup ==="
echo "Repo root: $REPO_ROOT"

# 1. Check Docker
echo "[1/5] Checking Docker..."
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not found. Please install Docker first."
    exit 1
fi
docker info >/dev/null 2>&1 || {
    echo "ERROR: Docker daemon not running or insufficient permissions."
    echo "Try: sudo usermod -aG docker \$USER  (then re-login)"
    exit 1
}
echo "  Docker OK"

# 2. Install Python dependencies
echo "[2/5] Installing Python dependencies..."
if command -v uv &>/dev/null; then
    echo "  Using uv..."
    cd "$REPO_ROOT"
    uv pip install -e ".[eval]" 2>/dev/null || uv pip install -e .
    uv pip install fastapi uvicorn httpx pyyaml
elif command -v pip &>/dev/null; then
    echo "  Using pip..."
    cd "$REPO_ROOT"
    pip install -e ".[eval]"
    pip install fastapi uvicorn httpx pyyaml
else
    echo "ERROR: Neither uv nor pip found. Please install Python 3.10+."
    exit 1
fi
echo "  Python deps OK"

# 3. Pull Docker images
echo "[3/5] Pulling Docker images (this may take a while)..."
bash "$REPO_ROOT/scripts/docker_pull_all.sh" "yimjk/ale-bench" || {
    echo "WARNING: docker_pull_all.sh failed. You may need to pull images manually."
    echo "  docker pull yimjk/ale-bench:cpp17-202301"
    echo "  docker pull rust:1.79.0-buster"
}
# Also pull the Rust build image
docker pull rust:1.79.0-buster || echo "WARNING: Failed to pull rust:1.79.0-buster"
echo "  Docker images OK"

# 4. Create cache directory
echo "[4/5] Creating cache directory..."
CACHE_DIR="${ALE_EVAL_CACHE:-/data/ale_bench_cache}"
mkdir -p "$CACHE_DIR/problems"
chmod -R 777 "$CACHE_DIR" 2>/dev/null || true
echo "  Cache dir: $CACHE_DIR"

# 5. Test import
echo "[5/5] Testing ale_bench import..."
python3 -c "import ale_bench; print('  ale_bench version:', ale_bench.__version__ if hasattr(ale_bench, '__version__') else 'OK')" || {
    echo "WARNING: ale_bench import failed. Check installation."
}

echo ""
echo "=== Setup complete ==="
echo "Next: bash $SCRIPT_DIR/start_devhost.sh"
