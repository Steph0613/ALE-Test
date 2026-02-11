# Remote Evaluator Deployment Guide

## Overview

The remote evaluator runs on the development host (10.214.54.87) where Docker is available.
It provides a FastAPI HTTP API that the training container calls to compile, run, and judge code.

## Prerequisites

- **Docker** installed and running (your user must be in the `docker` group)
- **Python 3.10+** with pip or uv
- **Disk space**: ~50GB+ for Docker images and problem caches
- **Network**: Port 8000 (configurable) accessible from the training container

## Setup Steps

### 1. Clone the repo (if not already done)

```bash
cd /path/to/ALE-Bench
```

### 2. Run setup script

```bash
bash experiments/grpo_ale_generalization/remote_evaluator/setup_devhost.sh
```

This will:
- Verify Docker is available
- Install Python dependencies (ale_bench + fastapi + uvicorn)
- Pull all required Docker images (~15GB total)
- Create the cache directory at `/data/ale_bench_cache/`

### 3. Start the evaluator

```bash
bash experiments/grpo_ale_generalization/remote_evaluator/start_devhost.sh
```

Or with custom settings:

```bash
export ALE_EVAL_PORT=8000
export ALE_EVAL_MAX_CONCURRENT=4
export ALE_EVAL_CACHE=/data/ale_bench_cache
export ALE_EVAL_API_KEY=""  # optional
bash experiments/grpo_ale_generalization/remote_evaluator/start_devhost.sh
```

### 4. Verify

```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","max_concurrent":"4"}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALE_EVAL_PORT` | 8000 | Server port |
| `ALE_EVAL_MAX_CONCURRENT` | 4 | Max concurrent Docker evaluations |
| `ALE_EVAL_CACHE` | /data/ale_bench_cache | Problem data cache directory |
| `ALE_EVAL_API_KEY` | (empty) | Optional API key for authentication |
| `ALE_EVAL_WALL_TIMEOUT` | 600 | Wall time timeout per evaluate call (seconds) |
| `ALE_BENCH_CACHE` | ~/.cache/ale-bench | HuggingFace data cache (used by ale_bench) |

## API Endpoints

### GET /health

Returns `{"status": "ok", "max_concurrent": "N"}`

### POST /warmup

Pre-downloads and builds tools for a problem.

```json
{"problem_id": "ahc001", "lite_version": true}
```

### POST /evaluate

Evaluates code on given seeds.

```json
{
  "problem_id": "ahc001",
  "code_language": "cpp17",
  "judge_version": "202301",
  "code": "#include <iostream>\nint main() { ... }",
  "seeds": [0, 1, 2],
  "lite_version": true
}
```

## Common Issues

### Docker permission denied

```bash
sudo usermod -aG docker $USER
# Then log out and log back in
```

### Port already in use

```bash
# Find process using port 8000
lsof -i :8000
# Or use a different port
export ALE_EVAL_PORT=8001
```

### First warmup is slow

The first time a problem is loaded:
1. Data is downloaded from HuggingFace (~10-100MB per problem)
2. Rust tools are compiled in Docker (~30-60s)

Subsequent calls use the cache.

### HuggingFace token

If you need authenticated access to the dataset:

```bash
export HF_TOKEN=hf_xxxxx
# Or: huggingface-cli login
```

### Docker images not found

```bash
# Pull all required images
bash scripts/docker_pull_all.sh yimjk/ale-bench
docker pull rust:1.79.0-buster
```

### Disk space

Check available space:
```bash
df -h /data
docker system df
```

Clean up:
```bash
docker system prune -f
```

## SSH Tunnel (if direct access is not available)

If the training container cannot directly reach port 8000, use SSH tunneling:

```bash
# From training container:
ssh -L 8000:localhost:8000 user@10.214.54.87 -N &
# Then set remote_evaluator_url: "http://localhost:8000" in exp.yaml
```

## Monitoring

```bash
# View logs
tail -f experiments/grpo_ale_generalization/remote_evaluator/evaluator.log

# Check running Docker containers
docker ps
```
