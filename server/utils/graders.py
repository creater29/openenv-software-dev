from __future__ import annotations

from typing import Any, Dict

# openenv validate walks the ENTIRE response tree and rejects any float
# that is <= 0.0 or >= 1.0.  Use these as the safe open-interval bounds.
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def clamp_score(value: float) -> float:
    """Clamp a single float to the open interval (0, 1)."""
    return max(_SCORE_MIN, min(_SCORE_MAX, value))


def sanitize_any(obj: Any) -> Any:
    """Recursively walk obj and clamp every float to (_SCORE_MIN, _SCORE_MAX).

    The openenv validator recurses into every nested dict and list in the
    JSON response, including reward.components and observation.metadata.weights.
    This function must be applied to BOTH the reward dict AND the full
    observation dict before they are returned in any response.
    """
    if isinstance(obj, dict):
        return {k: sanitize_any(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_any(v) for v in obj]
    if isinstance(obj, float):
        return clamp_score(obj)
    # int, str, bool, None pass through untouched
    return obj


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
    """Generic shaped reward shared across tasks, clamped to (0, 1)."""
    score = (
        (tests_passed_ratio * w_pass)
        + (improvement_over_last_step * w_improve)
        - (steps_taken * w_step_penalty)
        - destructive_action_penalty
    )
    return clamp_score(score)
