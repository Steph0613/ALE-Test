"""
Prompt builder for ALE-Bench GRPO iterative refinement.

Constructs prompts for:
  - Round 1: Initial code generation from problem statement
  - Round 2..R: Refinement with feedback from previous round

Controls prompt length to avoid context overflow:
  - Truncates best_code if exceeding max_code_chars
  - Truncates feedback messages
"""

from typing import Optional


# Maximum characters for code in feedback prompt
MAX_CODE_CHARS = 6000
# Maximum characters for error/feedback message
MAX_FEEDBACK_CHARS = 500
# Maximum characters for problem statement
MAX_STATEMENT_CHARS = 8000


def build_initial_prompt(
    statement: str,
    example_input: str,
    example_output: str,
    tool_readme: str,
    code_language: str = "cpp17",
    score_type: str = "maximize",
) -> list[dict[str, str]]:
    """Build the prompt for Round 1 (initial code generation).

    Args:
        statement: Problem statement text.
        example_input: Example input.
        example_output: Example output.
        tool_readme: Tool README with problem-specific constraints.
        code_language: Target language.
        score_type: "maximize" or "minimize".

    Returns:
        List of messages in OpenAI chat format [{"role": ..., "content": ...}].
    """
    lang_display = {
        "cpp17": "C++17",
        "cpp20": "C++20",
        "cpp23": "C++23",
        "python": "Python 3",
        "python3": "Python 3",
        "rust": "Rust",
    }.get(code_language, code_language)

    objective = "maximize" if score_type == "maximize" else "minimize"

    system_msg = (
        f"You are an expert competitive programmer. "
        f"Write a complete {lang_display} solution. "
        f"Output ONLY a single code block (```{_lang_tag(code_language)}\\n...\\n```). "
        f"No explanations, no comments outside the code."
    )

    # Truncate statement if too long
    stmt = statement[:MAX_STATEMENT_CHARS]
    if len(statement) > MAX_STATEMENT_CHARS:
        stmt += "\n\n[Statement truncated...]"

    user_msg = f"""## Problem

{stmt}

## Tool / Tester Information

{tool_readme[:3000]}

## Example

Input:
```
{example_input[:2000]}
```

Output:
```
{example_output[:2000]}
```

## Requirements

- Language: {lang_display}
- Read from stdin, write to stdout
- Objective: {objective} the score
- No debug output to stdout
- Output exactly ONE code block
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_refinement_prompt(
    statement: str,
    code_language: str,
    score_type: str,
    best_code: str,
    best_score: float,
    last_feedback: dict,
    round_num: int,
    total_rounds: int,
) -> list[dict[str, str]]:
    """Build the prompt for Round 2..R (refinement with feedback).

    Args:
        statement: Problem statement (may be truncated).
        code_language: Target language.
        score_type: "maximize" or "minimize".
        best_code: Current best code.
        best_score: Current best raw score.
        last_feedback: Dict with keys: fail_type, score, mean_time, mean_mem, short_message.
        round_num: Current round number (2-based).
        total_rounds: Total number of rounds.

    Returns:
        List of messages in OpenAI chat format.
    """
    lang_display = {
        "cpp17": "C++17",
        "cpp20": "C++20",
        "cpp23": "C++23",
        "python": "Python 3",
        "python3": "Python 3",
        "rust": "Rust",
    }.get(code_language, code_language)

    objective = "maximize" if score_type == "maximize" else "minimize"

    system_msg = (
        f"You are an expert competitive programmer. "
        f"You are improving your solution (round {round_num}/{total_rounds}). "
        f"Output ONLY a single code block with the COMPLETE improved solution. "
        f"No explanations."
    )

    # Truncate code for prompt
    code_display = best_code
    if len(best_code) > MAX_CODE_CHARS:
        # Keep first and last parts
        half = MAX_CODE_CHARS // 2
        code_display = best_code[:half] + "\n\n// ... [truncated] ...\n\n" + best_code[-half:]

    fail_type = last_feedback.get("fail_type", "OK")
    score = last_feedback.get("score", 0)
    mean_time = last_feedback.get("mean_time")
    mean_mem = last_feedback.get("mean_mem")
    short_msg = last_feedback.get("short_message", "")

    # Build feedback section
    feedback_lines = [
        f"- Status: {fail_type}",
        f"- Score: {score}",
    ]
    if mean_time is not None:
        feedback_lines.append(f"- Mean execution time: {mean_time:.3f}s")
    if mean_mem is not None:
        feedback_lines.append(f"- Mean memory: {mean_mem / 1024:.0f} KB")
    if short_msg:
        feedback_lines.append(f"- Message: {short_msg[:MAX_FEEDBACK_CHARS]}")

    feedback_str = "\n".join(feedback_lines)

    # Guidance based on fail type
    if fail_type == "COMPILE_ERROR":
        guidance = "Fix the compilation error. Check syntax and includes."
    elif fail_type == "RE":
        guidance = "Fix the runtime error. Check array bounds, null pointers, and edge cases."
    elif fail_type == "TLE":
        guidance = "Optimize for speed. Reduce time complexity or use more efficient algorithms."
    elif fail_type == "MLE":
        guidance = "Reduce memory usage. Use more memory-efficient data structures."
    elif fail_type == "WA":
        guidance = "Fix the logic error. The output format or algorithm may be incorrect."
    elif fail_type == "NO_CODE":
        guidance = "Your previous response did not contain valid code. Output a complete solution."
    else:
        guidance = f"Try to {'maximize' if objective == 'maximize' else 'minimize'} the score further."

    # Brief problem reminder (first 2000 chars)
    stmt_brief = statement[:2000]
    if len(statement) > 2000:
        stmt_brief += "\n[...]"

    user_msg = f"""## Problem (Brief)

{stmt_brief}

## Previous Evaluation Feedback

{feedback_str}

## Guidance

{guidance}

## Current Best Code (score={best_score})

```{_lang_tag(code_language)}
{code_display}
```

## Task

Improve the solution to {objective} the score. This is round {round_num} of {total_rounds}.
Output exactly ONE complete code block ({lang_display}).
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _lang_tag(code_language: str) -> str:
    """Get markdown language tag."""
    mapping = {
        "cpp17": "cpp",
        "cpp20": "cpp",
        "cpp23": "cpp",
        "python": "python",
        "python3": "python",
        "rust": "rust",
    }
    return mapping.get(code_language, code_language)
