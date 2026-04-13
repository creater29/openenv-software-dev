from __future__ import annotations

from typing import Any, Dict

# The openenv validator rejects any float that is exactly 0.0 or exactly 1.0
# anywhere in the JSON response — including nested dicts and lists.
# Every float we emit must live in the strictly open interval (0, 1).
_SCORE_MIN: float = 0.001   # replaces raw 0.0 / negative values
_SCORE_MAX: float = 0.999   # replaces raw 1.0 / values >= 1


def clamp_score(value: float) -> float:
    """Clamp *value* into the open interval (_SCORE_MIN, _SCORE_MAX).

    This is the single chokepoint that prevents 0.0 and 1.0 from ever
    entering a response.  Call it at every division site AND at every
    formula result — defence in depth.
    """
    return max(_SCORE_MIN, min(_SCORE_MAX, float(value)))


def safe_ratio(passed: int, total: int) -> float:
    """Convert a passed/total test count into a clamped (0, 1) float.

    This replaces the pattern ``0.0 if total == 0 else passed / total``
    that appeared in all three task files.  The raw division ``passed /
    total`` can yield exactly 1.0 when all tests pass; wrapping with
    clamp_score() ensures that never reaches the validator.
    """
    if total <= 0:
        return _SCORE_MIN           # no tests ran → minimum, not zero
    raw = passed / total            # may be 0.0 or 1.0 — clamped below
    return clamp_score(raw)


def sanitize_any(obj: Any) -> Any:
    """Recursively walk *obj* and clamp every float to (_SCORE_MIN, _SCORE_MAX).

    The openenv validator recurses into every nested dict and list in the
    JSON response, including reward.components and observation.metadata.weights.
    Apply this to BOTH the reward dict AND the full observation dict before
    returning any HTTP response.
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
    """Return a clamped penalty for destructive edits (deleting most code)."""
    if not files:
        return clamp_score(0.3)

    empties = 0
    tiny = 0
    for content in files.values():
        stripped = (content or '').strip()
        if not stripped:
            empties += 1
        elif len(stripped) < 20:
            tiny += 1

    ratio = (empties + 0.5 * tiny) / max(1, len(files))
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
    """Generic shaped reward clamped to the open interval (0, 1).

    Formula:
        reward = (tests_passed_ratio * W_pass)
               + (improvement_over_last_step * W_improve)
               - (steps_taken * W_step_penalty)
               - destructive_action_penalty

    All inputs are assumed to already be clamped; the final result is
    clamped again as a last-resort safety net.
    """
    score = (
        clamp_score(tests_passed_ratio) * w_pass
        + clamp_score(improvement_over_last_step) * w_improve
        - steps_taken * w_step_penalty
        - destructive_action_penalty
    )
    return clamp_score(score)
