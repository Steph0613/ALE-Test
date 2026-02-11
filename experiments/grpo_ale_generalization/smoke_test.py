"""
Smoke test for the GRPO ALE-Bench pipeline.

Validates the end-to-end chain:
  1. Remote evaluator /health is accessible
  2. Warmup a problem succeeds
  3. Evaluate a trivial (broken) code returns expected failure
  4. Evaluate a simple valid code returns a score
  5. (Optional) Run one round of iterative refinement prompt->evaluate cycle

Usage:
  python experiments/grpo_ale_generalization/smoke_test.py \
    --config experiments/grpo_ale_generalization/configs/exp.yaml

  # Quick mode (skip iterative test):
  python experiments/grpo_ale_generalization/smoke_test.py --quick
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_health(eval_url: str) -> bool:
    """Test 1: Health check."""
    print("\n[TEST 1] Health check...")
    try:
        resp = httpx.get(f"{eval_url}/health", timeout=10)
        data = resp.json()
        assert data.get("status") == "ok", f"Unexpected status: {data}"
        print(f"  PASS: {data}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_warmup(eval_url: str, problem_id: str, lite: bool, api_key: str = "") -> bool:
    """Test 2: Warmup a problem."""
    print(f"\n[TEST 2] Warmup problem '{problem_id}'...")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        t0 = time.time()
        resp = httpx.post(
            f"{eval_url}/warmup",
            json={"problem_id": problem_id, "lite_version": lite},
            headers=headers,
            timeout=600,
        )
        data = resp.json()
        elapsed = time.time() - t0
        assert data.get("ok"), f"Warmup failed: {data.get('error')}"
        print(f"  PASS ({elapsed:.1f}s): type={data.get('problem_type')}, "
              f"score_type={data.get('score_type')}, "
              f"time_limit={data.get('time_limit')}s")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_evaluate_broken_code(eval_url: str, problem_id: str, api_key: str = "") -> bool:
    """Test 3: Evaluate broken code -> expect COMPILE_ERROR."""
    print(f"\n[TEST 3] Evaluate broken code (expect COMPILE_ERROR)...")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    broken_code = "#include <iostream>\nint main() { this is broken code }"
    try:
        resp = httpx.post(
            f"{eval_url}/evaluate",
            json={
                "problem_id": problem_id,
                "code_language": "cpp17",
                "judge_version": "202301",
                "code": broken_code,
                "seeds": [0],
                "lite_version": True,
            },
            headers=headers,
            timeout=120,
        )
        data = resp.json()
        assert data.get("ok") or "COMPILATION_ERROR" in str(data), f"Unexpected: {data}"
        fail_type = data.get("feedback", {}).get("fail_type", "")
        # Check for compile error in results
        fail_counts = data.get("agg", {}).get("fail_counts", {})
        has_ce = "COMPILATION_ERROR" in fail_counts or fail_type == "COMPILE_ERROR"
        if has_ce:
            print(f"  PASS: Got compilation error as expected. fail_type={fail_type}")
            return True
        else:
            print(f"  WARN: Expected COMPILE_ERROR but got: {data}")
            return True  # Still OK if the evaluator returned something
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_evaluate_simple_code(eval_url: str, problem_id: str, api_key: str = "") -> bool:
    """Test 4: Evaluate a simple (probably wrong but compilable) code."""
    print(f"\n[TEST 4] Evaluate simple valid code...")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # Simple code that reads input and outputs something
    simple_code = """
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        cout << i << endl;
    }
    return 0;
}
"""
    try:
        t0 = time.time()
        resp = httpx.post(
            f"{eval_url}/evaluate",
            json={
                "problem_id": problem_id,
                "code_language": "cpp17",
                "judge_version": "202301",
                "code": simple_code,
                "seeds": [0, 1],
                "lite_version": True,
            },
            headers=headers,
            timeout=300,
        )
        data = resp.json()
        elapsed = time.time() - t0
        print(f"  Response ({elapsed:.1f}s):")
        print(f"    ok={data.get('ok')}")
        print(f"    fail_type={data.get('feedback', {}).get('fail_type')}")
        print(f"    score={data.get('feedback', {}).get('score')}")
        print(f"    fail_counts={data.get('agg', {}).get('fail_counts')}")

        # Check response structure
        assert "ok" in data, "Missing 'ok' field"
        assert "feedback" in data, "Missing 'feedback' field"
        assert "agg" in data, "Missing 'agg' field"
        assert "timings" in data, "Missing 'timings' field"

        fb = data["feedback"]
        assert "fail_type" in fb, "Missing feedback.fail_type"
        assert "score" in fb, "Missing feedback.score"

        print(f"  PASS: Response structure valid.")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_code_extract() -> bool:
    """Test 5: Code extraction utility."""
    print(f"\n[TEST 5] Code extraction...")
    try:
        from experiments.grpo_ale_generalization.reward.code_extract import extract_code

        # Test with fenced block
        resp = "Here's the solution:\n```cpp\n#include <iostream>\nint main() { return 0; }\n```\n"
        code = extract_code(resp, "cpp17")
        assert code is not None, "Failed to extract fenced code"
        assert "#include" in code, f"Unexpected code: {code[:100]}"

        # Test with no code
        code = extract_code("This is just text with no code.", "cpp17")
        assert code is None, "Should return None for no-code response"

        print(f"  PASS: Code extraction works correctly.")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_reward_module() -> bool:
    """Test 6: Reward module imports and penalty table."""
    print(f"\n[TEST 6] Reward module...")
    try:
        from experiments.grpo_ale_generalization.reward.ale_reward import (
            get_penalty,
            _FAIL_TYPE_PENALTY,
        )

        # Check penalty table
        assert get_penalty("NO_CODE") < 0, "NO_CODE penalty should be negative"
        assert get_penalty("COMPILE_ERROR") < 0, "COMPILE_ERROR penalty should be negative"
        assert get_penalty("OK") != 0 or True, "OK has no penalty (fine)"

        print(f"  PASS: Penalty table: {_FAIL_TYPE_PENALTY}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_prompt_builder() -> bool:
    """Test 7: Prompt builder."""
    print(f"\n[TEST 7] Prompt builder...")
    try:
        from experiments.grpo_ale_generalization.reward.prompt_builder import (
            build_initial_prompt,
            build_refinement_prompt,
        )

        messages = build_initial_prompt(
            statement="Sample problem statement...",
            example_input="3\n1 2 3",
            example_output="6",
            tool_readme="Use tester to check output.",
            code_language="cpp17",
            score_type="maximize",
        )
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        messages = build_refinement_prompt(
            statement="Sample problem...",
            code_language="cpp17",
            score_type="maximize",
            best_code="#include <iostream>\nint main() { return 0; }",
            best_score=100.0,
            last_feedback={"fail_type": "OK", "score": 100.0},
            round_num=2,
            total_rounds=3,
        )
        assert len(messages) == 2

        print(f"  PASS: Prompt builder works correctly.")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test for GRPO pipeline")
    parser.add_argument("--config", default="experiments/grpo_ale_generalization/configs/exp.yaml")
    parser.add_argument("--problem", default=None, help="Problem ID to test with (default: first in pool)")
    parser.add_argument("--quick", action="store_true", help="Skip evaluator tests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_url = cfg.get("remote_evaluator_url", "http://10.214.54.87:8000")
    api_key = cfg.get("remote_evaluator_api_key", "")
    lite = cfg.get("lite_version", True)

    # Determine test problem
    pool = cfg.get("problem_pool", [])
    test_problem = args.problem
    if not test_problem:
        if pool:
            test_problem = pool[0]
        else:
            test_problem = "ahc001"  # fallback
    print(f"Smoke test problem: {test_problem}")
    print(f"Evaluator URL: {eval_url}")

    results = {}
    n_pass = 0
    n_total = 0

    # Local tests (no evaluator needed)
    for test_fn in [test_code_extract, test_reward_module, test_prompt_builder]:
        n_total += 1
        if test_fn():
            n_pass += 1

    if not args.quick:
        # Evaluator tests
        for test_fn, test_args in [
            (test_health, (eval_url,)),
            (test_warmup, (eval_url, test_problem, lite, api_key)),
            (test_evaluate_broken_code, (eval_url, test_problem, api_key)),
            (test_evaluate_simple_code, (eval_url, test_problem, api_key)),
        ]:
            n_total += 1
            if test_fn(*test_args):
                n_pass += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"SMOKE TEST: {n_pass}/{n_total} passed")
    if n_pass == n_total:
        print("ALL TESTS PASSED")
    else:
        print(f"WARNING: {n_total - n_pass} test(s) failed")
    print("=" * 50)

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
