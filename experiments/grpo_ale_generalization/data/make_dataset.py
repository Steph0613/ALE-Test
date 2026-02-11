"""
Generate training dataset for GRPO ALE-Bench generalization experiment.

Reads exp.yaml to determine:
  - problem_pool / train_problem_ids / test_problem_ids
  - split_seed for reproducible train/test split
  - samples_per_problem

Outputs a JSONL (or Parquet) file where each row is one episode start:
  {
    "prompt": "<initial prompt for Round 1>",
    "problem_id": "ahcXXX",
    "extra_info": {
      "problem_id": "ahcXXX",
      "code_language": "cpp17",
      "judge_version": "202301",
      "score_type": "maximize",
      "problem_type": "batch",
      "iterative_refine_num_rounds": 3,
      "seeds": [0, 1, 2],
      "best_norm_before": 0.0
    }
  }

Usage:
  python experiments/grpo_ale_generalization/data/make_dataset.py \
    --config experiments/grpo_ale_generalization/configs/exp.yaml
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

# Add repo root to path so we can import ale_bench
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import httpx


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_split(cfg: dict) -> tuple[list[str], list[str]]:
    """Resolve train/test problem IDs from config.

    Returns:
        (train_ids, test_ids)
    """
    train_ids = cfg.get("train_problem_ids", [])
    test_ids = cfg.get("test_problem_ids", [])

    if train_ids and test_ids:
        print(f"Using explicit split: train={len(train_ids)}, test={len(test_ids)}")
        return train_ids, test_ids

    pool = cfg.get("problem_pool", [])
    num_train = cfg.get("num_train", 8)
    num_test = cfg.get("num_test", 4)

    if len(pool) < num_train + num_test:
        print(f"ERROR: problem_pool has {len(pool)} items, need at least {num_train + num_test}.")
        print("Run: python experiments/grpo_ale_generalization/tools/list_problems.py")
        print("Then fill problem_pool in exp.yaml with at least 12 problem IDs.")
        sys.exit(1)

    seed = cfg.get("split_seed", 42)
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    train_ids = shuffled[:num_train]
    test_ids = shuffled[num_train:num_train + num_test]

    print(f"Split (seed={seed}): train={train_ids}, test={test_ids}")
    return train_ids, test_ids


def get_problem_metadata(eval_url: str, problem_id: str, lite_version: bool, api_key: str = "") -> dict:
    """Warmup a problem and get its metadata from the remote evaluator."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.post(
            f"{eval_url}/warmup",
            json={"problem_id": problem_id, "lite_version": lite_version},
            headers=headers,
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            return data
        else:
            print(f"  WARNING: warmup for {problem_id} returned ok=false: {data.get('error')}")
            return data
    except Exception as e:
        print(f"  ERROR: Failed to warmup {problem_id}: {e}")
        return {"ok": False, "error": str(e)}


def build_initial_prompt_from_metadata(
    problem_id: str,
    code_language: str,
    score_type: str,
) -> str:
    """Build a minimal initial prompt when we can't fetch the full problem statement.

    In the actual training loop, the reward function will load the full problem data
    from the evaluator. This prompt is a placeholder to start the episode.
    """
    lang_map = {"cpp17": "C++17", "python": "Python 3", "cpp20": "C++20", "cpp23": "C++23"}
    lang = lang_map.get(code_language, code_language)
    objective = "maximize" if score_type == "maximize" else "minimize"

    return (
        f"You are solving competitive programming problem '{problem_id}'. "
        f"Write a complete {lang} solution that reads from stdin and writes to stdout. "
        f"Your goal is to {objective} the score. "
        f"Output ONLY a single code block."
    )


def build_initial_prompt_full(
    statement: str,
    example_input: str,
    example_output: str,
    tool_readme: str,
    code_language: str,
    score_type: str,
) -> str:
    """Build a full initial prompt from problem data."""
    # Import the prompt builder
    from experiments.grpo_ale_generalization.reward.prompt_builder import build_initial_prompt
    messages = build_initial_prompt(
        statement=statement,
        example_input=example_input,
        example_output=example_output,
        tool_readme=tool_readme,
        code_language=code_language,
        score_type=score_type,
    )
    # Flatten to single string for dataset
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    return "\n\n".join(parts)


def try_load_problem_local(problem_id: str, lite_version: bool) -> dict | None:
    """Try to load problem data locally using ale_bench.data.load_problem."""
    try:
        from ale_bench.data import load_problem
        problem, seeds, standings, rpm, data_root = load_problem(problem_id, lite_version)
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
    except Exception as e:
        print(f"  Note: Could not load {problem_id} locally ({e}), using minimal prompt.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO training dataset")
    parser.add_argument("--config", default="experiments/grpo_ale_generalization/configs/exp.yaml")
    parser.add_argument("--train_only", action="store_true", help="Only generate for train problems")
    parser.add_argument("--no_evaluator", action="store_true", help="Don't call remote evaluator for warmup")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_ids, test_ids = resolve_split(cfg)

    code_language = cfg.get("code_language", "cpp17")
    judge_version = cfg.get("judge_version", "202301")
    lite_version = cfg.get("lite_version", True)
    eval_url = cfg.get("remote_evaluator_url", "http://10.214.54.87:8000")
    api_key = cfg.get("remote_evaluator_api_key", "")
    dataset_cfg = cfg.get("dataset", {})
    samples_per_problem = dataset_cfg.get("samples_per_problem", 4)
    output_dir = Path(dataset_cfg.get("output_dir", "experiments/grpo_ale_generalization/data/train_data"))
    output_format = dataset_cfg.get("output_format", "jsonl")
    iterative_refine_num_rounds = cfg.get("iterative_refine_num_rounds", 3)
    train_k_seeds = cfg.get("train_k_seeds", 3)

    # Determine which problem IDs to include
    problem_ids = list(train_ids)
    if not args.train_only:
        problem_ids.extend(test_ids)

    print(f"\nGenerating dataset for {len(problem_ids)} problems, {samples_per_problem} samples each")

    # Warmup all problems (if evaluator is available)
    metadata_cache: dict[str, dict] = {}
    if not args.no_evaluator:
        print("\nWarming up problems on remote evaluator...")
        for pid in problem_ids:
            print(f"  Warming up {pid}...")
            meta = get_problem_metadata(eval_url, pid, lite_version, api_key)
            metadata_cache[pid] = meta

    # Generate dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    for pid in problem_ids:
        is_train = pid in train_ids
        split = "train" if is_train else "test"
        meta = metadata_cache.get(pid, {})
        score_type = meta.get("score_type", "maximize")
        problem_type = meta.get("problem_type", "batch")

        # Try to build full prompt with local data
        local_data = try_load_problem_local(pid, lite_version)
        if local_data:
            score_type = local_data["score_type"]
            problem_type = local_data["problem_type"]

        for sample_idx in range(samples_per_problem):
            # Determine seeds for this sample
            if local_data:
                public_seeds = local_data["seeds_public"]
            else:
                public_seeds = list(range(100))

            # Select K seeds for training evaluation
            rng = random.Random(42 + sample_idx)
            if len(public_seeds) >= train_k_seeds:
                selected_seeds = rng.sample(public_seeds, train_k_seeds)
            else:
                selected_seeds = public_seeds[:train_k_seeds]

            # Build prompt
            if local_data:
                prompt = build_initial_prompt_full(
                    statement=local_data["statement"],
                    example_input=local_data["example_input"],
                    example_output=local_data["example_output"],
                    tool_readme=local_data["tool_readme"],
                    code_language=code_language,
                    score_type=score_type,
                )
            else:
                prompt = build_initial_prompt_from_metadata(pid, code_language, score_type)

            extra_info = {
                "problem_id": pid,
                "code_language": code_language,
                "judge_version": judge_version,
                "score_type": score_type,
                "problem_type": problem_type,
                "iterative_refine_num_rounds": iterative_refine_num_rounds,
                "seeds": selected_seeds,
                "best_norm_before": 0.0,
                "split": split,
                "sample_idx": sample_idx,
            }

            sample = {
                "prompt": prompt,
                "problem_id": pid,
                "extra_info": json.dumps(extra_info),
            }
            samples.append(sample)

    # Write output
    if output_format == "jsonl":
        output_path = output_dir / "dataset.jsonl"
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    elif output_format == "parquet":
        try:
            import pandas as pd
            df = pd.DataFrame(samples)
            output_path = output_dir / "dataset.parquet"
            df.to_parquet(output_path, index=False)
        except ImportError:
            print("WARNING: pandas not available, falling back to jsonl")
            output_path = output_dir / "dataset.jsonl"
            with open(output_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Also write split info
    split_info = {
        "train_problem_ids": train_ids,
        "test_problem_ids": test_ids,
        "split_seed": cfg.get("split_seed", 42),
        "num_samples": len(samples),
        "samples_per_problem": samples_per_problem,
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nDataset written to {output_path}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Train problems: {train_ids}")
    print(f"  Test problems: {test_ids}")
    print(f"  Split info: {output_dir / 'split_info.json'}")


if __name__ == "__main__":
    main()
