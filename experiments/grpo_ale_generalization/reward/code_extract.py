"""
Code extraction utilities for ALE-Bench GRPO training.

Extracts code blocks from LLM responses with robust fallback logic.

Rules (in order):
  1. Find all fenced code blocks (```...```)
  2. If any block has a language tag matching the target (cpp, python, etc.),
     return the LAST such block (models tend to refine at the end).
  3. Otherwise, return the LONGEST fenced block.
  4. If no fenced blocks, attempt to treat the entire response as code
     (only if it looks like code: contains main/include/#include/def/import).
  5. If nothing works, return None (NO_CODE).
"""

import re
from typing import Optional


# Pattern matches ```lang\n...\n``` with optional language tag
_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\s*\n(.*?)```",
    re.DOTALL,
)

# Language aliases
_LANG_ALIASES = {
    "cpp17": {"cpp", "c++", "cpp17", "cxx"},
    "cpp20": {"cpp", "c++", "cpp20", "cxx"},
    "cpp23": {"cpp", "c++", "cpp23", "cxx"},
    "python": {"python", "python3", "py"},
    "rust": {"rust", "rs"},
}


def extract_code(response: str, target_language: str = "cpp17") -> Optional[str]:
    """Extract code from an LLM response.

    Args:
        response: The raw LLM response text.
        target_language: Target code language (e.g., "cpp17", "python").

    Returns:
        Extracted code string, or None if no valid code found.
    """
    if not response or not response.strip():
        return None

    # Find all fenced code blocks
    blocks = _CODE_BLOCK_RE.findall(response)
    # blocks is list of (lang_tag, code_content)

    if not blocks:
        # Fallback: try entire response as code if it looks like code
        return _try_raw_code(response, target_language)

    # Get language aliases for target
    aliases = _LANG_ALIASES.get(target_language, {target_language})

    # Filter blocks matching target language
    matching = [(lang, code) for lang, code in blocks if lang.lower() in aliases]

    if matching:
        # Return the LAST matching block (model tends to refine toward end)
        return matching[-1][1].strip()

    # No language-tagged match: return the LONGEST block
    longest = max(blocks, key=lambda x: len(x[1]))
    return longest[1].strip()


def _try_raw_code(response: str, target_language: str) -> Optional[str]:
    """Try to interpret the raw response as code."""
    text = response.strip()

    # Heuristic: check for common code patterns
    code_indicators = {
        "cpp17": ["#include", "int main", "using namespace", "scanf", "printf", "cin", "cout"],
        "cpp20": ["#include", "int main", "using namespace"],
        "cpp23": ["#include", "int main", "using namespace"],
        "python": ["import ", "def ", "print(", "input(", "sys.stdin", "from "],
        "rust": ["fn main", "use std", "let ", "println!"],
    }

    indicators = code_indicators.get(target_language, [])
    if any(ind in text for ind in indicators):
        return text

    return None


def sanitize_code(code: str, target_language: str = "cpp17") -> str:
    """Sanitize extracted code by removing common artifacts.

    Args:
        code: Extracted code string.
        target_language: Target language.

    Returns:
        Cleaned code string.
    """
    # Remove leading/trailing whitespace
    code = code.strip()

    # Remove any remaining markdown artifacts
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]  # remove opening ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code
