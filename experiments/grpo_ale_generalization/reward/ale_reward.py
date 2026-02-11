"""
ALE-Bench GRPO Reward Function.

This module implements the custom reward function for VeRL GRPO training.
It handles:
  1. Code extraction from model responses
  2. Communication with the remote evaluator
  3. Per-problem reward normalization (z-score with EMA)
  4. Failure penalty assignment
  5. Incremental reward computation for iterative refinement

=== REWARD FORMULA ===

For each round t of iterative refinement:

  raw_score_t = mean absolute score across K seeds (from evaluator)

  If failure (NO_CODE, COMPILE_ERROR, RE, TLE, MLE, WA):
    reward_t = penalty[fail_type]  (see PENALTY TABLE below)

  If success (OK):
    norm_score_t = zscore_clip(raw_score_t)  # per-problem z-score, clipped to [-3, 3]

    If incremental mode:
      reward_t = clip(norm_score_after - norm_score_before, [-3, 3])
    Else:
      reward_t = norm_score_t

  Optional time penalty: reward_t -= time_penalty_coeff * mean_time / time_limit

=== PENALTY TABLE ===

  Failure Type      | Default Penalty
  ------------------|----------------
  NO_CODE           | -2.0
  COMPILE_ERROR     | -1.5
  RUNTIME_ERROR     | -1.0
  TIME_LIMIT_EXCEEDED | -1.0
  MEMORY_LIMIT_EXCEEDED | -1.0
  WRONG_ANSWER      | -0.5

=== Z-SCORE NORMALIZATION (per problem_id) ===

  Maintains running EMA of mean and variance per problem:
    ema_mean = (1 - alpha) * ema_mean + alpha * raw_score
    ema_var  = (1 - alpha) * ema_var  + alpha * (raw_score - ema_mean)^2
    zscore   = (raw_score - ema_mean) / max(sqrt(ema_var), 1e-6)
    clipped  = clip(zscore, [clip_min, clip_max])

  For MINIMIZE problems, raw_score is negated before normalization
  so that "better" always means "higher normalized score".
"""

import json
import logging
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Optional

import httpx
import yaml

from experiments.grpo_ale_generalization.reward.code_extract import extract_code, sanitize_code

logger = logging.getLogger("ale_reward")

# ---------------------------------------------------------------------------
# Load config (once at import time)
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.environ.get(
    "ALE_EXP_CONFIG",
    "experiments/grpo_ale_generalization/configs/exp.yaml",
)


def _load_config() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_CONFIG = _load_config()

# Reward config
_REWARD_CFG = _CONFIG.get("reward", {})
_PENALTY = _REWARD_CFG.get("penalty", {})
_EMA_ALPHA = _REWARD_CFG.get("ema_alpha", 0.1)
_CLIP_MIN = _REWARD_CFG.get("clip_min", -3.0)
_CLIP_MAX = _REWARD_CFG.get("clip_max", 3.0)
_INCREMENTAL = _REWARD_CFG.get("incremental", True)
_TIME_PENALTY_COEFF = _REWARD_CFG.get("time_penalty_coeff", 0.0)
_NORMALIZATION = _REWARD_CFG.get("normalization", "zscore_ema")

# Remote evaluator
_EVAL_URL = _CONFIG.get("remote_evaluator_url", "http://10.214.54.87:8000")
_EVAL_TIMEOUT = _CONFIG.get("remote_evaluator_timeout", 300)
_EVAL_API_KEY = _CONFIG.get("remote_evaluator_api_key", "")

# Code language
_CODE_LANGUAGE = _CONFIG.get("code_language", "cpp17")
_JUDGE_VERSION = _CONFIG.get("judge_version", "202301")
_LITE_VERSION = _CONFIG.get("lite_version", True)

# Seeds
_TRAIN_K_SEEDS = _CONFIG.get("train_k_seeds", 3)


# ---------------------------------------------------------------------------
# Per-problem EMA statistics for z-score normalization
# ---------------------------------------------------------------------------
class ProblemStats:
    """Running EMA statistics for a single problem_id."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.mean: float = 0.0
        self.var: float = 1.0
        self.initialized: bool = False
        self.lock = Lock()
        # Diagnostics
        self.n_updates: int = 0
        self.total_raw: float = 0.0
        self.total_failures: int = 0

    def update_and_normalize(self, raw_score: float) -> float:
        """Update EMA stats and return z-score clipped to [clip_min, clip_max].

        Args:
            raw_score: Raw score (already direction-adjusted: higher=better).

        Returns:
            Clipped z-score.
        """
        with self.lock:
            self.n_updates += 1
            self.total_raw += raw_score

            if not self.initialized:
                self.mean = raw_score
                self.var = 1.0  # start with unit variance
                self.initialized = True
            else:
                self.mean = (1 - self.alpha) * self.mean + self.alpha * raw_score
                diff = raw_score - self.mean
                self.var = (1 - self.alpha) * self.var + self.alpha * diff * diff

            std = max(self.var ** 0.5, 1e-6)
            zscore = (raw_score - self.mean) / std
            return max(_CLIP_MIN, min(_CLIP_MAX, zscore))

    def record_failure(self) -> None:
        with self.lock:
            self.total_failures += 1

    def get_diagnostics(self) -> dict[str, Any]:
        with self.lock:
            return {
                "n_updates": self.n_updates,
                "ema_mean": self.mean,
                "ema_std": self.var ** 0.5,
                "mean_raw": self.total_raw / max(self.n_updates, 1),
                "total_failures": self.total_failures,
                "fail_rate": self.total_failures / max(self.n_updates + self.total_failures, 1),
            }


# Global stats registry
_problem_stats: dict[str, ProblemStats] = defaultdict(lambda: ProblemStats(_EMA_ALPHA))
_stats_lock = Lock()


def get_problem_stats(problem_id: str) -> ProblemStats:
    with _stats_lock:
        return _problem_stats[problem_id]


# ---------------------------------------------------------------------------
# Remote evaluator client
# ---------------------------------------------------------------------------
def evaluate_code(
    problem_id: str,
    code: str,
    seeds: list[int],
    code_language: str = _CODE_LANGUAGE,
    judge_version: str = _JUDGE_VERSION,
    lite_version: bool = _LITE_VERSION,
) -> dict[str, Any]:
    """Call the remote evaluator to evaluate code.

    Returns the evaluator response dict.
    """
    headers = {}
    if _EVAL_API_KEY:
        headers["Authorization"] = f"Bearer {_EVAL_API_KEY}"

    payload = {
        "problem_id": problem_id,
        "code_language": code_language,
        "judge_version": judge_version,
        "code": code,
        "seeds": seeds,
        "skip_local_visualization": True,
        "return_details": False,
        "lite_version": lite_version,
    }

    try:
        with httpx.Client(timeout=_EVAL_TIMEOUT) as client:
            resp = client.post(f"{_EVAL_URL}/evaluate", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Evaluate call failed for {problem_id}: {e}")
        return {
            "ok": False,
            "error": str(e),
            "feedback": {"fail_type": "INTERNAL_ERROR", "score": 0},
            "agg": {"mean_abs_score": 0, "sum_abs_score": 0, "fail_counts": {}},
        }


# ---------------------------------------------------------------------------
# Penalty mapping
# ---------------------------------------------------------------------------
_FAIL_TYPE_PENALTY = {
    "NO_CODE": _PENALTY.get("NO_CODE", -2.0),
    "COMPILE_ERROR": _PENALTY.get("COMPILE_ERROR", -1.5),
    "RE": _PENALTY.get("RUNTIME_ERROR", -1.0),
    "TLE": _PENALTY.get("TIME_LIMIT_EXCEEDED", -1.0),
    "MLE": _PENALTY.get("MEMORY_LIMIT_EXCEEDED", -1.0),
    "WA": _PENALTY.get("WRONG_ANSWER", -0.5),
    "INTERNAL_ERROR": _PENALTY.get("COMPILE_ERROR", -1.5),  # treat as compile error
}


def get_penalty(fail_type: str) -> float:
    """Get penalty for a failure type."""
    return _FAIL_TYPE_PENALTY.get(fail_type, -1.0)


# ---------------------------------------------------------------------------
# Direction adjustment for MINIMIZE problems
# ---------------------------------------------------------------------------
def adjust_direction(raw_score: float, score_type: str) -> float:
    """Adjust score direction so that higher = better.

    For MINIMIZE problems, negate the score.
    """
    if score_type == "minimize":
        return -raw_score
    return raw_score


# ---------------------------------------------------------------------------
# Main reward computation
# ---------------------------------------------------------------------------
def compute_single_reward(
    problem_id: str,
    code: Optional[str],
    seeds: list[int],
    score_type: str,
    best_norm_before: float = 0.0,
    code_language: str = _CODE_LANGUAGE,
    judge_version: str = _JUDGE_VERSION,
) -> tuple[float, dict[str, Any]]:
    """Compute reward for a single code submission.

    Args:
        problem_id: Problem identifier.
        code: Extracted code (None means NO_CODE).
        seeds: Seeds to evaluate on.
        score_type: "maximize" or "minimize".
        best_norm_before: Normalized best score before this round.
        code_language: Code language.
        judge_version: Judge version.

    Returns:
        Tuple of (reward, info_dict).
    """
    stats = get_problem_stats(problem_id)
    info: dict[str, Any] = {"problem_id": problem_id}

    # NO_CODE case
    if code is None or not code.strip():
        penalty = get_penalty("NO_CODE")
        stats.record_failure()
        info.update({"fail_type": "NO_CODE", "reward": penalty, "raw_score": 0})
        return penalty, info

    # Evaluate
    result = evaluate_code(problem_id, code, seeds, code_language, judge_version)
    info["eval_result"] = {
        "ok": result.get("ok"),
        "fail_type": result.get("feedback", {}).get("fail_type", "INTERNAL_ERROR"),
        "mean_abs_score": result.get("agg", {}).get("mean_abs_score", 0),
    }

    if not result.get("ok", False):
        fail_type = result.get("feedback", {}).get("fail_type", "INTERNAL_ERROR")
        penalty = get_penalty(fail_type)
        stats.record_failure()
        info.update({"fail_type": fail_type, "reward": penalty, "raw_score": 0})
        return penalty, info

    feedback = result.get("feedback", {})
    fail_type = feedback.get("fail_type", "OK")

    if fail_type != "OK":
        penalty = get_penalty(fail_type)
        stats.record_failure()
        info.update({
            "fail_type": fail_type,
            "reward": penalty,
            "raw_score": result.get("agg", {}).get("mean_abs_score", 0),
        })
        return penalty, info

    # Successful evaluation
    raw_score = result.get("agg", {}).get("mean_abs_score", 0.0)
    adjusted = adjust_direction(raw_score, score_type)

    if _NORMALIZATION == "zscore_ema":
        norm_score = stats.update_and_normalize(adjusted)
    elif _NORMALIZATION == "raw_clip":
        norm_score = max(_CLIP_MIN, min(_CLIP_MAX, adjusted))
    else:
        norm_score = adjusted

    # Incremental or absolute reward
    if _INCREMENTAL:
        reward = max(_CLIP_MIN, min(_CLIP_MAX, norm_score - best_norm_before))
    else:
        reward = norm_score

    # Time penalty
    mean_time = feedback.get("mean_time")
    if _TIME_PENALTY_COEFF > 0 and mean_time is not None:
        # Assume default time limit ~5s if not available
        reward -= _TIME_PENALTY_COEFF * mean_time / 5.0

    info.update({
        "fail_type": "OK",
        "reward": reward,
        "raw_score": raw_score,
        "adjusted_score": adjusted,
        "norm_score": norm_score,
        "best_norm_before": best_norm_before,
        "mean_time": mean_time,
    })

    return reward, info


# ---------------------------------------------------------------------------
# VeRL-compatible reward function interface
# ---------------------------------------------------------------------------
def compute_reward(
    prompts: list[str],
    responses: list[str],
    extra_infos: list[dict[str, Any]],
) -> list[float]:
    """Compute rewards for a batch of (prompt, response) pairs.

    This is the entry point called by VeRL GRPO trainer.

    For each sample, the extra_info dict must contain:
      - problem_id: str
      - score_type: str ("maximize" or "minimize")
      - seeds: list[int] (seeds to evaluate on)

    Optionally:
      - code_language: str (default from config)
      - judge_version: str (default from config)
      - best_norm_before: float (for incremental reward)

    Args:
        prompts: List of prompt strings.
        responses: List of model response strings.
        extra_infos: List of extra info dicts.

    Returns:
        List of reward floats.
    """
    rewards = []
    all_infos = []

    for prompt, response, extra in zip(prompts, responses, extra_infos):
        problem_id = extra["problem_id"]
        score_type = extra.get("score_type", "maximize")
        seeds = extra.get("seeds", list(range(_TRAIN_K_SEEDS)))
        code_language = extra.get("code_language", _CODE_LANGUAGE)
        judge_version = extra.get("judge_version", _JUDGE_VERSION)
        best_norm_before = extra.get("best_norm_before", 0.0)

        # Extract code
        lang_for_extract = code_language.replace("python", "python")
        code = extract_code(response, lang_for_extract)
        if code:
            code = sanitize_code(code, lang_for_extract)

        reward, info = compute_single_reward(
            problem_id=problem_id,
            code=code,
            seeds=seeds,
            score_type=score_type,
            best_norm_before=best_norm_before,
            code_language=code_language,
            judge_version=judge_version,
        )

        rewards.append(reward)
        all_infos.append(info)

    # Log diagnostics periodically
    _log_diagnostics(all_infos)

    return rewards


# ---------------------------------------------------------------------------
# Diagnostic logging
# ---------------------------------------------------------------------------
_log_counter = 0
_LOG_EVERY = int(os.environ.get("ALE_REWARD_LOG_EVERY", "1"))


def _log_diagnostics(infos: list[dict[str, Any]]) -> None:
    """Log per-problem aggregated diagnostics."""
    global _log_counter
    _log_counter += 1
    if _log_counter % _LOG_EVERY != 0:
        return

    # Aggregate by problem
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for info in infos:
        by_problem[info.get("problem_id", "unknown")].append(info)

    for pid, pinfos in by_problem.items():
        rewards = [i["reward"] for i in pinfos]
        fail_types = [i.get("fail_type", "unknown") for i in pinfos]
        raw_scores = [i.get("raw_score", 0) for i in pinfos]
        n = len(pinfos)
        n_ok = sum(1 for ft in fail_types if ft == "OK")
        mean_reward = sum(rewards) / n if n > 0 else 0.0
        mean_raw = sum(raw_scores) / n if n > 0 else 0.0
        fail_rate = 1 - (n_ok / n) if n > 0 else 1.0

        stats = get_problem_stats(pid)
        diag = stats.get_diagnostics()

        logger.info(
            f"[Step {_log_counter}] problem={pid} "
            f"batch_size={n} mean_reward={mean_reward:.3f} "
            f"mean_raw_score={mean_raw:.1f} fail_rate={fail_rate:.2f} "
            f"ema_mean={diag['ema_mean']:.1f} ema_std={diag['ema_std']:.2f} "
            f"cumul_fail_rate={diag['fail_rate']:.2f}"
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Reward module loaded successfully.")
    print(f"Config: {_CONFIG_PATH}")
    print(f"Evaluator URL: {_EVAL_URL}")
    print(f"Penalty table: {_FAIL_TYPE_PENALTY}")
    print(f"Normalization: {_NORMALIZATION}, EMA alpha: {_EMA_ALPHA}")
    print(f"Clip range: [{_CLIP_MIN}, {_CLIP_MAX}]")
    print(f"Incremental: {_INCREMENTAL}")
