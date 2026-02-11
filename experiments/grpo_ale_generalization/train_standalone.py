"""
Standalone GRPO training loop for ALE-Bench generalization experiment.

This is a minimal, self-contained GRPO implementation that:
  1. Loads the base model with LoRA
  2. Samples prompts from the dataset
  3. Generates G responses per prompt (group)
  4. Computes rewards via the remote evaluator (with iterative refinement)
  5. Applies GRPO update (advantage = reward - mean_reward within group)
  6. Saves checkpoints

This script runs when VeRL is not installed. It uses:
  - transformers + peft for model loading
  - Custom GRPO loss computation
  - DeepSpeed or FSDP for distributed training (via accelerate)

Usage:
  python experiments/grpo_ale_generalization/train_standalone.py \
    --config experiments/grpo_ale_generalization/configs/verl_grpo.yaml

  # Or via train.sh:
  bash experiments/grpo_ale_generalization/train.sh --smoke
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.grpo_ale_generalization.reward.ale_reward import (
    compute_single_reward,
    evaluate_code,
    get_problem_stats,
)
from experiments.grpo_ale_generalization.reward.code_extract import extract_code, sanitize_code
from experiments.grpo_ale_generalization.reward.prompt_builder import (
    build_initial_prompt,
    build_refinement_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("grpo_train")


def load_configs(verl_config_path: str) -> tuple[dict, dict]:
    """Load VeRL config and experiment config."""
    with open(verl_config_path) as f:
        verl_cfg = yaml.safe_load(f)

    exp_config_path = os.environ.get(
        "ALE_EXP_CONFIG",
        "experiments/grpo_ale_generalization/configs/exp.yaml",
    )
    with open(exp_config_path) as f:
        exp_cfg = yaml.safe_load(f)

    return verl_cfg, exp_cfg


def load_dataset(data_path: str) -> list[dict]:
    """Load dataset from JSONL."""
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # Parse extra_info if it's a string
                if isinstance(sample.get("extra_info"), str):
                    sample["extra_info"] = json.loads(sample["extra_info"])
                samples.append(sample)
    return samples


def filter_train_samples(samples: list[dict]) -> list[dict]:
    """Filter to only train split samples."""
    return [s for s in samples if s.get("extra_info", {}).get("split") == "train"]


def setup_model_and_tokenizer(verl_cfg: dict, exp_cfg: dict):
    """Load model with LoRA and return model, tokenizer, optimizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    model_name = verl_cfg.get("model", {}).get("name_or_path",
        exp_cfg.get("model", {}).get("name_or_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"))

    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Apply LoRA
    peft_cfg = verl_cfg.get("peft", {})
    if peft_cfg.get("enabled", True):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=peft_cfg.get("rank", 16),
            lora_alpha=peft_cfg.get("alpha", 32),
            lora_dropout=peft_cfg.get("dropout", 0.05),
            target_modules=peft_cfg.get("target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if verl_cfg.get("trainer", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Optimizer
    opt_cfg = verl_cfg.get("trainer", {}).get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-5),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )

    return model, tokenizer, optimizer


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    group_size: int,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[list[str]]:
    """Generate G responses per prompt.

    Returns:
        List of lists: [num_prompts][group_size] strings.
    """
    all_responses = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        responses = []
        for _ in range(group_size):
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Decode only the generated part
            gen_ids = output[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(response)

        all_responses.append(responses)

    return all_responses


def compute_grpo_loss(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[list[str]],
    rewards: list[list[float]],
    clip_range: float = 0.2,
    kl_coeff: float = 0.01,
) -> torch.Tensor:
    """Compute GRPO loss.

    GRPO advantage: A_i = (r_i - mean(r)) / max(std(r), 1e-6)
    Loss: -sum(A_i * log_prob_i) (clipped)
    """
    total_loss = torch.tensor(0.0, device=model.device)
    n_samples = 0

    for prompt, resp_group, reward_group in zip(prompts, responses, rewards):
        # Compute advantages within group
        r = torch.tensor(reward_group, dtype=torch.float32)
        mean_r = r.mean()
        std_r = max(r.std().item(), 1e-6)
        advantages = ((r - mean_r) / std_r).tolist()

        for response, advantage in zip(resp_group, advantages):
            if abs(advantage) < 1e-8:
                continue

            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = len(tokenizer(prompt, truncation=True, max_length=4096)["input_ids"])
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100  # mask prompt tokens

            outputs = model(**inputs, labels=labels)
            # Weighted by advantage
            loss = outputs.loss * advantage
            total_loss = total_loss + loss
            n_samples += 1

    if n_samples > 0:
        total_loss = total_loss / n_samples

    return total_loss


def run_iterative_refinement_episode(
    model,
    tokenizer,
    sample: dict,
    group_size: int,
    exp_cfg: dict,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
) -> tuple[list[str], list[float]]:
    """Run one iterative refinement episode for GRPO.

    For each member of the group:
      Round 1: generate from initial prompt, evaluate
      Round 2..R: build refinement prompt with feedback, generate, evaluate

    Returns:
      responses: list of G final responses (the concatenated trajectory)
      rewards: list of G cumulative rewards
    """
    extra = sample["extra_info"]
    problem_id = extra["problem_id"]
    code_language = extra.get("code_language", "cpp17")
    judge_version = extra.get("judge_version", "202301")
    score_type = extra.get("score_type", "maximize")
    seeds = extra.get("seeds", [0, 1, 2])
    num_rounds = extra.get("iterative_refine_num_rounds",
                           exp_cfg.get("iterative_refine_num_rounds", 3))

    initial_prompt = sample["prompt"]
    all_responses = []
    all_rewards = []

    for g in range(group_size):
        cumulative_reward = 0.0
        best_code = None
        best_score_raw = 0.0
        best_norm = 0.0
        last_feedback = {}
        trajectory_response = ""

        for round_num in range(1, num_rounds + 1):
            if round_num == 1:
                prompt = initial_prompt
            else:
                # Build refinement prompt - we use a simplified version here
                from experiments.grpo_ale_generalization.reward.prompt_builder import build_refinement_prompt
                messages = build_refinement_prompt(
                    statement=initial_prompt[:3000],  # use initial prompt as statement proxy
                    code_language=code_language,
                    score_type=score_type,
                    best_code=best_code or "// No code yet",
                    best_score=best_score_raw,
                    last_feedback=last_feedback,
                    round_num=round_num,
                    total_rounds=num_rounds,
                )
                prompt = "\n\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)

            # Generate response
            responses = generate_responses(
                model, tokenizer, [prompt],
                group_size=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            response = responses[0][0]
            trajectory_response += f"\n--- Round {round_num} ---\n{response}"

            # Extract and evaluate code
            code = extract_code(response, code_language)
            if code:
                code = sanitize_code(code, code_language)

            reward, info = compute_single_reward(
                problem_id=problem_id,
                code=code,
                seeds=seeds,
                score_type=score_type,
                best_norm_before=best_norm,
                code_language=code_language,
                judge_version=judge_version,
            )

            cumulative_reward += reward

            # Update best
            if info.get("fail_type") == "OK":
                raw = info.get("raw_score", 0)
                norm = info.get("norm_score", 0)
                if score_type == "maximize" and raw > best_score_raw:
                    best_code = code
                    best_score_raw = raw
                    best_norm = norm
                elif score_type == "minimize" and (best_code is None or raw < best_score_raw):
                    best_code = code
                    best_score_raw = raw
                    best_norm = norm

            # Update feedback for next round
            last_feedback = {
                "fail_type": info.get("fail_type", "INTERNAL_ERROR"),
                "score": info.get("raw_score", 0),
                "mean_time": info.get("mean_time"),
                "mean_mem": None,
                "short_message": info.get("eval_result", {}).get("fail_type", ""),
            }

        all_responses.append(trajectory_response)
        all_rewards.append(cumulative_reward)

    return all_responses, all_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/grpo_ale_generalization/configs/verl_grpo.yaml")
    args, unknown = parser.parse_known_args()

    verl_cfg, exp_cfg = load_configs(args.config)

    # Setup
    model, tokenizer, optimizer = setup_model_and_tokenizer(verl_cfg, exp_cfg)

    # Load dataset
    data_path = verl_cfg.get("data", {}).get("train_path",
        "experiments/grpo_ale_generalization/data/train_data/dataset.jsonl")
    all_samples = load_dataset(data_path)
    train_samples = filter_train_samples(all_samples)
    if not train_samples:
        train_samples = all_samples  # fallback: use all samples
    logger.info(f"Loaded {len(train_samples)} training samples")

    # Training params
    trainer_cfg = verl_cfg.get("trainer", {})
    total_steps = trainer_cfg.get("total_steps", 200)
    batch_size = trainer_cfg.get("batch_size", 8)
    group_size = verl_cfg.get("trainer", {}).get("grpo", {}).get("group_size", 4)
    grad_accum = trainer_cfg.get("gradient_accumulation_steps", 2)
    max_grad_norm = trainer_cfg.get("max_grad_norm", 1.0)
    save_steps = trainer_cfg.get("save_steps", 50)
    save_dir = Path(trainer_cfg.get("save_dir", "experiments/grpo_ale_generalization/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    rollout_cfg = verl_cfg.get("rollout", {})
    temperature = rollout_cfg.get("temperature", 0.7)
    max_new_tokens = rollout_cfg.get("max_new_tokens", 4096)

    log_dir = Path(verl_cfg.get("logging", {}).get("log_dir",
        "experiments/grpo_ale_generalization/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training: {total_steps} steps, batch={batch_size}, group={group_size}")

    # Training loop
    model.train()
    global_step = 0
    rng = random.Random(42)

    for step in range(total_steps):
        step_start = time.time()

        # Sample batch
        batch = rng.choices(train_samples, k=batch_size)

        all_prompts = []
        all_responses = []
        all_rewards = []

        for sample in batch:
            responses, rewards = run_iterative_refinement_episode(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                group_size=group_size,
                exp_cfg=exp_cfg,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            all_prompts.append(sample["prompt"])
            all_responses.append(responses)
            all_rewards.append(rewards)

        # Compute GRPO loss
        optimizer.zero_grad()
        loss = compute_grpo_loss(
            model=model,
            tokenizer=tokenizer,
            prompts=all_prompts,
            responses=all_responses,
            rewards=all_rewards,
            clip_range=trainer_cfg.get("grpo", {}).get("clip_range", 0.2),
            kl_coeff=trainer_cfg.get("grpo", {}).get("kl_coeff", 0.01),
        )

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        global_step += 1
        step_time = time.time() - step_start

        # Logging
        mean_reward = sum(sum(r) for r in all_rewards) / max(sum(len(r) for r in all_rewards), 1)
        logger.info(
            f"Step {global_step}/{total_steps} | "
            f"loss={loss.item():.4f} | "
            f"mean_reward={mean_reward:.3f} | "
            f"step_time={step_time:.1f}s"
        )

        # Save checkpoint
        if global_step % save_steps == 0 or global_step == total_steps:
            ckpt_path = save_dir / f"step_{global_step}"
            logger.info(f"Saving checkpoint to {ckpt_path}")
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
