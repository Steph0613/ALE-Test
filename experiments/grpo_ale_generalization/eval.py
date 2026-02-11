"""
Evaluation script for GRPO ALE-Bench generalization experiment.

Compares base model vs finetuned checkpoint on test problems.
Uses iterative refinement (same as training) to generate trajectories.

Usage:
  # Evaluate both base and finetuned
  python experiments/grpo_ale_generalization/eval.py \
    --config experiments/grpo_ale_generalization/configs/exp.yaml \
    --ckpt experiments/grpo_ale_generalization/checkpoints/step_200

  # Base model only
  python experiments/grpo_ale_generalization/eval.py \
    --config experiments/grpo_ale_generalization/configs/exp.yaml \
    --base_only

Outputs:
  - results.json: Detailed per-problem results
  - results.md: Markdown table for reporting
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.grpo_ale_generalization.reward.code_extract import extract_code, sanitize_code
from experiments.grpo_ale_generalization.reward.prompt_builder import (
    build_initial_prompt,
    build_refinement_prompt,
)

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_split(cfg: dict) -> tuple[list[str], list[str]]:
    train_ids = cfg.get("train_problem_ids", [])
    test_ids = cfg.get("test_problem_ids", [])
    if train_ids and test_ids:
        return train_ids, test_ids
    pool = cfg.get("problem_pool", [])
    num_train = cfg.get("num_train", 8)
    num_test = cfg.get("num_test", 4)
    if len(pool) < num_train + num_test:
        logger.error(f"problem_pool too small: {len(pool)}")
        sys.exit(1)
    seed = cfg.get("split_seed", 42)
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[:num_train], shuffled[num_train:num_train + num_test]


def load_model(model_path: str, ckpt_path: Optional[str] = None):
    """Load model, optionally with LoRA checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if ckpt_path:
        logger.info(f"Loading LoRA checkpoint: {ckpt_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def warmup_problem(eval_url: str, problem_id: str, lite: bool, api_key: str = "") -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.post(
        f"{eval_url}/warmup",
        json={"problem_id": problem_id, "lite_version": lite},
        headers=headers,
        timeout=600,
    )
    return resp.json()


def evaluate_code(eval_url: str, problem_id: str, code: str, seeds: list[int],
                  code_language: str, judge_version: str, lite: bool, api_key: str = "") -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.post(
        f"{eval_url}/evaluate",
        json={
            "problem_id": problem_id,
            "code_language": code_language,
            "judge_version": judge_version,
            "code": code,
            "seeds": seeds,
            "skip_local_visualization": True,
            "return_details": False,
            "lite_version": lite,
        },
        headers=headers,
        timeout=300,
    )
    return resp.json()


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str,
                      max_new_tokens: int = 4096,
                      temperature: float = 0.7) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def try_load_problem_local(problem_id: str, lite: bool) -> Optional[dict]:
    """Load problem data locally."""
    try:
        from ale_bench.data import load_problem
        problem, seeds, _, _, _ = load_problem(problem_id, lite)
        return {
            "statement": problem.statement,
            "example_input": problem.example_input,
            "example_output": problem.example_output,
            "tool_readme": problem.tool_readme,
            "score_type": problem.metadata.score_type.value,
            "problem_type": problem.metadata.problem_type.value,
            "seeds_public": seeds.public,
            "seeds_private": seeds.private,
        }
    except Exception:
        return None


def evaluate_problem_with_model(
    model,
    tokenizer,
    problem_id: str,
    cfg: dict,
    eval_url: str,
    api_key: str,
    num_rounds: int,
    eval_seeds: list[int],
    code_language: str,
    judge_version: str,
    lite: bool,
    n_runs: int = 1,
) -> dict:
    """Evaluate a single problem with iterative refinement.

    Returns dict with best_score, fail_type, per_round scores, etc.
    """
    # Load problem data
    local_data = try_load_problem_local(problem_id, lite)
    score_type = "maximize"
    if local_data:
        score_type = local_data["score_type"]

    best_overall_score = 0.0
    run_results = []

    for run_idx in range(n_runs):
        best_code = None
        best_score_raw = 0.0
        last_feedback: dict[str, Any] = {}
        round_scores = []

        for round_num in range(1, num_rounds + 1):
            # Build prompt
            if round_num == 1:
                if local_data:
                    messages = build_initial_prompt(
                        statement=local_data["statement"],
                        example_input=local_data["example_input"],
                        example_output=local_data["example_output"],
                        tool_readme=local_data["tool_readme"],
                        code_language=code_language,
                        score_type=score_type,
                    )
                    prompt = "\n\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
                else:
                    prompt = f"Solve problem '{problem_id}'. Write complete {code_language} code."
            else:
                messages = build_refinement_prompt(
                    statement=(local_data["statement"][:2000] if local_data
                              else f"Problem: {problem_id}"),
                    code_language=code_language,
                    score_type=score_type,
                    best_code=best_code or "// No code yet",
                    best_score=best_score_raw,
                    last_feedback=last_feedback,
                    round_num=round_num,
                    total_rounds=num_rounds,
                )
                prompt = "\n\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)

            # Generate
            response = generate_response(model, tokenizer, prompt)
            code = extract_code(response, code_language)
            if code:
                code = sanitize_code(code, code_language)

            if not code:
                round_scores.append({"round": round_num, "fail_type": "NO_CODE", "score": 0})
                last_feedback = {"fail_type": "NO_CODE", "score": 0, "short_message": "No code extracted"}
                continue

            # Evaluate
            result = evaluate_code(eval_url, problem_id, code, eval_seeds,
                                   code_language, judge_version, lite, api_key)

            if not result.get("ok"):
                ft = result.get("feedback", {}).get("fail_type", "INTERNAL_ERROR")
                round_scores.append({"round": round_num, "fail_type": ft, "score": 0})
                last_feedback = {"fail_type": ft, "score": 0,
                                "short_message": result.get("error", "")[:200]}
                continue

            feedback = result.get("feedback", {})
            fail_type = feedback.get("fail_type", "OK")
            mean_score = result.get("agg", {}).get("mean_abs_score", 0.0)

            round_scores.append({
                "round": round_num,
                "fail_type": fail_type,
                "score": mean_score,
            })

            # Update best
            if fail_type == "OK":
                is_better = False
                if score_type == "maximize" and mean_score > best_score_raw:
                    is_better = True
                elif score_type == "minimize" and (best_code is None or mean_score < best_score_raw):
                    is_better = True

                if is_better:
                    best_code = code
                    best_score_raw = mean_score

            last_feedback = {
                "fail_type": fail_type,
                "score": mean_score,
                "mean_time": feedback.get("mean_time"),
                "mean_mem": feedback.get("mean_mem"),
                "short_message": feedback.get("short_message", "")[:200],
            }

        run_results.append({
            "run_idx": run_idx,
            "best_score": best_score_raw,
            "round_scores": round_scores,
        })
        best_overall_score = max(best_overall_score, best_score_raw) if score_type == "maximize" \
            else min(best_overall_score, best_score_raw) if best_overall_score != 0 else best_score_raw

    # Compute mean best across runs
    mean_best = sum(r["best_score"] for r in run_results) / max(len(run_results), 1)
    fail_rate = sum(
        1 for r in run_results
        if all(rs["fail_type"] != "OK" for rs in r["round_scores"])
    ) / max(len(run_results), 1)

    return {
        "problem_id": problem_id,
        "score_type": score_type,
        "mean_best_score": mean_best,
        "best_overall_score": best_overall_score,
        "fail_rate": fail_rate,
        "n_runs": n_runs,
        "run_results": run_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs finetuned model")
    parser.add_argument("--config", default="experiments/grpo_ale_generalization/configs/exp.yaml")
    parser.add_argument("--ckpt", default=None, help="Path to finetuned checkpoint")
    parser.add_argument("--base_only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of evaluation runs per problem")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_ids, test_ids = resolve_split(cfg)

    eval_cfg = cfg.get("eval", {})
    eval_url = cfg.get("remote_evaluator_url", "http://10.214.54.87:8000")
    api_key = cfg.get("remote_evaluator_api_key", "")
    code_language = cfg.get("code_language", "cpp17")
    judge_version = cfg.get("judge_version", "202301")
    lite = cfg.get("lite_version", True)
    num_rounds = eval_cfg.get("iterative_refine_num_rounds", cfg.get("iterative_refine_num_rounds", 3))
    eval_n_seeds = eval_cfg.get("eval_n_seeds", cfg.get("eval_n_seeds", 5))
    model_name = cfg.get("model", {}).get("name_or_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ckpt_path = args.ckpt or eval_cfg.get("checkpoint_path") or None
    output_dir = Path(eval_cfg.get("output_dir", "experiments/grpo_ale_generalization/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify evaluator
    try:
        health = httpx.get(f"{eval_url}/health", timeout=5).json()
        logger.info(f"Evaluator health: {health}")
    except Exception as e:
        logger.error(f"Cannot reach evaluator at {eval_url}: {e}")
        sys.exit(1)

    # Warmup test problems
    logger.info(f"Warming up {len(test_ids)} test problems...")
    for pid in test_ids:
        logger.info(f"  Warmup {pid}...")
        warmup_problem(eval_url, pid, lite, api_key)

    # Determine eval seeds
    # Use public seeds from first problem as reference
    eval_seeds = list(range(eval_n_seeds))  # fallback
    local_data = try_load_problem_local(test_ids[0], lite) if test_ids else None
    if local_data:
        pub = local_data["seeds_public"]
        eval_seeds = pub[:eval_n_seeds]
    logger.info(f"Eval seeds: {eval_seeds}")

    results: dict[str, Any] = {
        "config": {
            "test_problems": test_ids,
            "train_problems": train_ids,
            "model": model_name,
            "checkpoint": ckpt_path,
            "num_rounds": num_rounds,
            "eval_seeds": eval_seeds,
            "n_runs": args.n_runs,
        },
        "base": {},
        "finetuned": {},
    }

    # Evaluate BASE model
    logger.info("\n=== Evaluating BASE model ===")
    base_model, tokenizer = load_model(model_name)

    for pid in test_ids:
        logger.info(f"\n--- Base: {pid} ---")
        # Load problem-specific seeds
        pd = try_load_problem_local(pid, lite)
        seeds = pd["seeds_public"][:eval_n_seeds] if pd else eval_seeds

        result = evaluate_problem_with_model(
            model=base_model, tokenizer=tokenizer,
            problem_id=pid, cfg=cfg, eval_url=eval_url, api_key=api_key,
            num_rounds=num_rounds, eval_seeds=seeds,
            code_language=code_language, judge_version=judge_version, lite=lite,
            n_runs=args.n_runs,
        )
        results["base"][pid] = result
        logger.info(f"  Base {pid}: mean_best={result['mean_best_score']:.1f}, "
                     f"fail_rate={result['fail_rate']:.2f}")

    # Cleanup base model
    del base_model
    torch.cuda.empty_cache()

    # Evaluate FINETUNED model (if checkpoint provided)
    if not args.base_only and ckpt_path:
        logger.info("\n=== Evaluating FINETUNED model ===")
        ft_model, tokenizer = load_model(model_name, ckpt_path)

        for pid in test_ids:
            logger.info(f"\n--- FT: {pid} ---")
            pd = try_load_problem_local(pid, lite)
            seeds = pd["seeds_public"][:eval_n_seeds] if pd else eval_seeds

            result = evaluate_problem_with_model(
                model=ft_model, tokenizer=tokenizer,
                problem_id=pid, cfg=cfg, eval_url=eval_url, api_key=api_key,
                num_rounds=num_rounds, eval_seeds=seeds,
                code_language=code_language, judge_version=judge_version, lite=lite,
                n_runs=args.n_runs,
            )
            results["finetuned"][pid] = result
            logger.info(f"  FT {pid}: mean_best={result['mean_best_score']:.1f}, "
                         f"fail_rate={result['fail_rate']:.2f}")

        del ft_model
        torch.cuda.empty_cache()

    # Compute summary
    base_scores = [v["mean_best_score"] for v in results["base"].values()]
    base_mean = sum(base_scores) / max(len(base_scores), 1)

    ft_scores = [v["mean_best_score"] for v in results["finetuned"].values()] if results["finetuned"] else []
    ft_mean = sum(ft_scores) / max(len(ft_scores), 1) if ft_scores else 0.0
    delta = ft_mean - base_mean if ft_scores else 0.0

    results["summary"] = {
        "base_mean_score": base_mean,
        "ft_mean_score": ft_mean,
        "delta": delta,
        "num_test_problems": len(test_ids),
    }

    # Save results.json
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults written to {json_path}")

    # Generate results.md
    md_lines = [
        "# GRPO ALE-Bench Generalization Results",
        "",
        f"**Model**: {model_name}",
        f"**Checkpoint**: {ckpt_path or 'N/A'}",
        f"**Eval rounds**: {num_rounds}, **Seeds**: {eval_n_seeds}",
        "",
        "## Test Problems",
        "",
        "| Problem | Score Type | Base Mean Best | FT Mean Best | Delta | Base Fail% | FT Fail% |",
        "|---------|-----------|---------------|-------------|-------|-----------|---------|",
    ]

    for pid in test_ids:
        b = results["base"].get(pid, {})
        f_res = results["finetuned"].get(pid, {})
        b_score = b.get("mean_best_score", 0)
        f_score = f_res.get("mean_best_score", 0) if f_res else "-"
        d = f_score - b_score if isinstance(f_score, (int, float)) else "-"
        b_fail = f"{b.get('fail_rate', 0) * 100:.0f}%"
        f_fail = f"{f_res.get('fail_rate', 0) * 100:.0f}%" if f_res else "-"
        s_type = b.get("score_type", "?")

        md_lines.append(
            f"| {pid} | {s_type} | {b_score:.1f} | "
            f"{'%.1f' % f_score if isinstance(f_score, (int, float)) else f_score} | "
            f"{'%.1f' % d if isinstance(d, (int, float)) else d} | "
            f"{b_fail} | {f_fail} |"
        )

    md_lines.extend([
        "",
        "## Overall",
        "",
        f"- **Base mean score**: {base_mean:.1f}",
        f"- **FT mean score**: {ft_mean:.1f}" if ft_scores else "- **FT**: Not evaluated",
        f"- **Test mean delta**: {delta:.1f}" if ft_scores else "",
    ])

    md_path = output_dir / "results.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown results written to {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test problems: {test_ids}")
    print(f"Base mean score: {base_mean:.1f}")
    if ft_scores:
        print(f"FT mean score: {ft_mean:.1f}")
        print(f"Delta (FT - Base): {delta:+.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
