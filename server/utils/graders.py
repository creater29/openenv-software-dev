from __future__ import annotations

from typing import Dict

# openenv validate requires every score field to be strictly inside (0, 1).
# Exact 0.0 and 1.0 are both rejected. We use epsilon boundaries.
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def clamp_score(value: float, minimum: float = _SCORE_MIN, maximum: float = _SCORE_MAX) -> float:
    """Clamp value to the open interval (0, 1) required by openenv validate."""
    return max(minimum, min(maximum, value))


def sanitize_reward_dict(reward: Dict[str, float]) -> Dict[str, float]:
    """Ensure every top-level numeric field in a reward dict is strictly in (0, 1).

    The openenv validator checks ALL numeric fields in the RewardModel, not just
    'reward'. Fields like tests_passed_ratio, improvement_over_last_step,
    step_penalty, and destructive_action_penalty must all satisfy 0 < x < 1.

    Negative values (e.g. improvement when score drops) and values > 1 are all
    clamped to [_SCORE_MIN, _SCORE_MAX]. The 'components' sub-dict is left as-is
    because the validator does not appear to recurse into it.
    """
    sanitized = {}
    for key, value in reward.items():
        if key == "components":
            # Leave component breakdown untouched — validator ignores sub-dicts
            sanitized[key] = value
        elif isinstance(value, float):
            sanitized[key] = clamp_score(value)
        else:
            sanitized[key] = value
    return sanitized


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
