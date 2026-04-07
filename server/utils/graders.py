from __future__ import annotations

from typing import Dict


def clamp_score(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
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
