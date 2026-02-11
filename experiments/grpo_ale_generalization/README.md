# GRPO ALE-Bench Generalization Experiment

## What This Experiment Answers

**Question**: After RL (GRPO) training on a set of open-ended algorithmic problems, does the model generalize to unseen problems?

**Design**:
- Pool of 12 ALE-Bench problems, split into 8 train / 4 test
- Train: GRPO with iterative refinement on train problems
- Evaluate: Compare base model vs finetuned model on test problems
- Metric: Mean best score delta (finetuned - base) on test set

## Architecture

```
Training Container (8x H800)          Dev Host (10.214.54.87)
+---------------------------+          +----------------------+
| Model + LoRA              |  HTTP    | FastAPI Evaluator    |
| GRPO Trainer              |--------->| Docker (compile/run) |
| Reward Function           |  /eval   | Problem Cache        |
| Dataset                   |          |                      |
+---------------------------+          +----------------------+
```

- **Training container**: Runs the model, generates code, computes GRPO updates
- **Dev host**: Runs Docker containers for compilation, execution, and judging
- Communication via HTTP (no Docker-in-Docker on training side)

## Quick Start

### 1. Deploy evaluator (on dev host 10.214.54.87)

```bash
bash experiments/grpo_ale_generalization/remote_evaluator/setup_devhost.sh
bash experiments/grpo_ale_generalization/remote_evaluator/start_devhost.sh
```

### 2. Choose problems (on training container)

```bash
# List available problems
python experiments/grpo_ale_generalization/tools/list_problems.py

# Edit exp.yaml: fill in 12 problem IDs
vi experiments/grpo_ale_generalization/configs/exp.yaml
```

### 3. Smoke test

```bash
# Quick (local tests only, no evaluator needed)
python experiments/grpo_ale_generalization/smoke_test.py --quick

# Full (needs evaluator running)
python experiments/grpo_ale_generalization/smoke_test.py
```

### 4. Generate dataset

```bash
python experiments/grpo_ale_generalization/data/make_dataset.py \
  --config experiments/grpo_ale_generalization/configs/exp.yaml
```

### 5. Train

```bash
# Smoke test (4 steps, minimal config)
bash experiments/grpo_ale_generalization/train.sh --smoke

# Main training
bash experiments/grpo_ale_generalization/train.sh
```

### 6. Evaluate

```bash
# Base model only
python experiments/grpo_ale_generalization/eval.py \
  --config experiments/grpo_ale_generalization/configs/exp.yaml \
  --base_only

# Base + finetuned
python experiments/grpo_ale_generalization/eval.py \
  --config experiments/grpo_ale_generalization/configs/exp.yaml \
  --ckpt experiments/grpo_ale_generalization/checkpoints/step_200
```

## Configuration

All experiment settings are in **one file**: `configs/exp.yaml`

To switch problems, change only:
```yaml
problem_pool:
  - ahc001
  - ahc002
  # ... 12 total
```

The split is automatic (seed=42, first 8 train, last 4 test), or you can specify explicitly:
```yaml
train_problem_ids: [ahc001, ahc002, ...]
test_problem_ids: [ahc009, ahc010, ...]
```

## Reward Design

### Formula

For each round t of iterative refinement:

```
If failure (NO_CODE, COMPILE_ERROR, RE, TLE, MLE, WA):
    reward_t = penalty[fail_type]

If success (OK):
    raw_score = mean absolute score across K seeds
    adjusted  = -raw_score if MINIMIZE, else raw_score  (direction normalization)
    norm      = zscore_ema_clip(adjusted)                (per-problem z-score)
    reward_t  = clip(norm_after - norm_before, [-3, 3])  (incremental)
```

### Penalty Table

| Failure Type | Default Penalty |
|-------------|----------------|
| NO_CODE | -2.0 |
| COMPILE_ERROR | -1.5 |
| RUNTIME_ERROR (RE) | -1.0 |
| TIME_LIMIT_EXCEEDED (TLE) | -1.0 |
| MEMORY_LIMIT_EXCEEDED (MLE) | -1.0 |
| WRONG_ANSWER (WA) | -0.5 |

### Z-Score Normalization (per problem)

```
ema_mean = (1 - alpha) * ema_mean + alpha * adjusted_score
ema_var  = (1 - alpha) * ema_var  + alpha * (adjusted_score - ema_mean)^2
zscore   = (adjusted_score - ema_mean) / max(sqrt(ema_var), 1e-6)
clipped  = clip(zscore, [-3, 3])
```

**Why this works**:
- EMA adapts to different score scales across problems
- Direction adjustment handles MINIMIZE problems
- Clipping keeps rewards in a stable range for RL
- Incremental mode teaches the model to improve, not just generate

### Diagnostics

The reward function logs per-problem stats every step:
```
[Step 10] problem=ahc001 batch_size=4 mean_reward=0.523 mean_raw_score=12345.6 fail_rate=0.25 ema_mean=12000.0 ema_std=500.00 cumul_fail_rate=0.15
```

## Iterative Refinement

Each episode has R rounds (default 3):

1. **Round 1**: Model generates code from problem statement -> evaluate -> get feedback
2. **Round 2..R**: Feedback + best code -> model improves -> evaluate -> update best

The trajectory maintains:
- `best_code`: highest-scoring code so far
- `best_score_raw`: raw score of best code
- `last_feedback`: fail_type, score, time, memory, error message

Each round produces one GRPO sample: `(prompt=current state, response=new code, reward=incremental score change)`

## Training Configuration

### Smoke Config (fast verification)
- 4 steps, batch=2, group=2, R=2 rounds, K=1 seed
- ~10 min to verify the pipeline works

### Main Config
- 200 steps, batch=8, group=4, R=3 rounds, K=3 seeds
- LoRA rank=16, bf16, gradient checkpointing
- 8x H800 with FSDP

### Bottleneck: Remote Evaluator

The main bottleneck is Docker execution on the dev host. Tips:
- Increase `ALE_EVAL_MAX_CONCURRENT` on the dev host (default 4)
- Increase `client_max_concurrent` in exp.yaml (default 4)
- Use `lite_version: true` for faster evaluation
- Reduce `train_k_seeds` (fewer seeds per evaluation)
- Warmup all problems before training starts (make_dataset.py does this)

## Directory Structure

```
experiments/grpo_ale_generalization/
├── configs/
│   ├── exp.yaml              # Main experiment config (change this!)
│   └── verl_grpo.yaml        # VeRL training config
├── data/
│   ├── make_dataset.py       # Dataset generation script
│   └── train_data/           # Generated dataset (gitignored)
├── reward/
│   ├── ale_reward.py         # VeRL custom reward function
│   ├── code_extract.py       # Code extraction from LLM responses
│   └── prompt_builder.py     # Prompt construction for iterative refinement
├── remote_evaluator/
│   ├── evaluator_server.py   # FastAPI evaluator service
│   ├── setup_devhost.sh      # Dev host setup script
│   ├── start_devhost.sh      # Dev host start script
│   └── README_devhost.md     # Dev host deployment guide
├── tools/
│   └── list_problems.py      # List available problem IDs
├── train.sh                  # Training entry point
├── train_standalone.py       # Standalone GRPO trainer (no VeRL dep)
├── eval.py                   # Evaluation script
├── smoke_test.py             # Pipeline smoke test
└── README.md                 # This file
```

## Common Issues

### "Remote evaluator not reachable"
- Ensure the evaluator is running on 10.214.54.87:8000
- Check firewall rules
- Try SSH tunnel: `ssh -L 8000:localhost:8000 user@10.214.54.87 -N &`

### "Docker images not found" (on dev host)
```bash
bash scripts/docker_pull_all.sh yimjk/ale-bench
docker pull rust:1.79.0-buster
```

### "problem_pool is empty"
```bash
python experiments/grpo_ale_generalization/tools/list_problems.py
# Then edit configs/exp.yaml with 12 problem IDs
```

### Training is very slow
- Bottleneck is usually remote evaluation
- Check evaluator logs: `tail -f remote_evaluator/evaluator.log`
- Increase concurrency on both sides
- Use lite_version for faster eval (fewer test cases per seed)

### Out of GPU memory
- Reduce batch_size or group_size
- Ensure gradient_checkpointing is enabled
- Reduce max_new_tokens
- Use lower LoRA rank

## Verification Criteria

The experiment is successful if:
- [x] `make_dataset` -> `train` (few steps) -> `eval` (base+ft) completes end-to-end
- [x] `eval` output contains per-test-problem base/ft comparison with overall delta
- [x] Changing only `exp.yaml` switches all 12 problems and the 8/4 split
- [x] All evaluation during training goes through the remote evaluator (no Docker in training container)
- [x] Reward normalization and penalty diagnostics visible in training logs
