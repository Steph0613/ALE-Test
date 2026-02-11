"""
List all available ALE-Bench problem IDs.

Helps you choose 12 problems for exp.yaml problem_pool.

Usage:
  python experiments/grpo_ale_generalization/tools/list_problems.py
  python experiments/grpo_ale_generalization/tools/list_problems.py --lite
  python experiments/grpo_ale_generalization/tools/list_problems.py --details
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ale_bench.data import list_problem_ids


def main():
    parser = argparse.ArgumentParser(description="List available ALE-Bench problems")
    parser.add_argument("--lite", action="store_true", help="List lite version problem IDs")
    parser.add_argument("--details", action="store_true",
                       help="Load each problem and show metadata (slow, requires HF download)")
    args = parser.parse_args()

    print("Fetching problem IDs...")
    ids = list_problem_ids(lite_version=args.lite)
    print(f"\nFound {len(ids)} problems{'(lite)' if args.lite else ''}:\n")

    if args.details:
        from ale_bench.data import load_problem
        print(f"{'ID':<30} {'Type':<10} {'Score':<10} {'Time(s)':<8} {'Mem(B)':<12}")
        print("-" * 80)
        for pid in ids:
            try:
                problem, seeds, _, _, _ = load_problem(pid, lite_version=True)
                print(f"{pid:<30} {problem.metadata.problem_type.value:<10} "
                      f"{problem.metadata.score_type.value:<10} "
                      f"{problem.constraints.time_limit:<8.1f} "
                      f"{problem.constraints.memory_limit:<12}")
            except Exception as e:
                print(f"{pid:<30} ERROR: {e}")
    else:
        for pid in ids:
            print(f"  {pid}")

    print(f"\nTotal: {len(ids)} problems")
    print("\nTo use in exp.yaml, copy 12 IDs into problem_pool:")
    print("  problem_pool:")
    for pid in ids[:12]:
        print(f"    - {pid}")
    print("  # ... (showing first 12 as example)")


if __name__ == "__main__":
    main()
