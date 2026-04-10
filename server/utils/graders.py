from __future__ import annotations

from typing import Dict

# openenv validate requires scores to be strictly inside the open interval (0, 1).
# Exact 0.0 and 1.0 are rejected, so we use a small epsilon to keep values
# safely away from both boundaries while preserving the full semantic range.
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def clamp_score(value: float, minimum: float = _SCORE_MIN, maximum: float = _SCORE_MAX) -> float:
    """Clamp *value* to the open interval (0, 1) required by openenv validate.

    The defaults enforce the strict boundary: no score ever equals exactly 0.0
    or 1.0.  Callers that previously passed explicit minimum/maximum of 0.0/1.0
    should remove those keyword arguments so the safe defaults apply.
    """
    return max(minimum, min(maximum, value))


def compute_destructive_penalty(files: Dict[str, str]) -> float:
    """Penalize destructive edits like deleting most code across files."""

    if not files:
        return 0.3

    empties = 0
    tiny = 0
    for content in files.values():
        stripped = (content or "").strip()
        if not stripped:
            empties += 1
        elif len(stripped) < 20:
            tiny += 1

    ratio = (empties + (0.5 * tiny)) / max(1, len(files))
    return clamp_score(0.3 * ratio)


def compute_shaped_reward(
    tests_passed_ratio: float,
    improvement_over_last_step: float,
    steps_taken: int,
    destructive_action_penalty: float,
    w_pass: float,
    w_improve: float,
    w_step_penalty: float,
) -> float:
    """Generic shaped reward shared across tasks.

    reward = (tests_passed_ratio * W_pass)
           + (improvement_over_last_step * W_improve)
           - (steps_taken * W_step_penalty)
           - (destructive_action_penalty)
    """

    score = (
        (tests_passed_ratio * w_pass)
        + (improvement_over_last_step * w_improve)
        - (steps_taken * w_step_penalty)
        - destructive_action_penalty
    )
    return clamp_score(score)
