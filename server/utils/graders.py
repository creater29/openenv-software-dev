from __future__ import annotations

from typing import Any, Dict

# ── Interval constants ────────────────────────────────────────────────────────
# The openenv validator rejects ANY float that is exactly 0.0 or exactly 1.0
# anywhere in the JSON response, including deeply nested dicts and lists.
# Every float we emit MUST live in the strictly open interval (0, 1).
_SCORE_MIN: float = 0.001   # floor  — replaces 0.0 / negatives
_SCORE_MAX: float = 0.999   # ceiling — replaces 1.0 / values above 1


# ── Primary guard ─────────────────────────────────────────────────────────────

def guard_score(value: float) -> float:
    """Explicit if/else gate: map 0 → 0.001 and 1 → 0.999, clamp everything else.

    This is the canonical if/else guard required at EVERY site that produces
    or consumes a score float.  Using a named function (not an inline ternary)
    makes the intent unambiguous and easy to grep.

    Logic:
        if value is 0.0 (or below)  → return 0.001   (never emit 0.0)
        elif value is 1.0 (or above) → return 0.999  (never emit 1.0)
        else                          → return value  (interior values unchanged)
    """
    v = float(value)
    if v <= 0.0:        # catches exact 0.0, negatives, -inf, -nan-likes
        return _SCORE_MIN
    elif v >= 1.0:      # catches exact 1.0, values above 1, +inf
        return _SCORE_MAX
    else:               # strictly interior — pass through unchanged
        return v


# Alias so existing call-sites that use clamp_score() continue to work.
# Both names now call the same if/else logic.
clamp_score = guard_score


def safe_ratio(passed: int, total: int) -> float:
    """Convert a passed/total count into a guarded (0, 1) float.

    This is THE single division site for all test-count ratios.  The raw
    division ``passed / total`` can yield exactly 0.0 (nothing passed) or
    exactly 1.0 (everything passed) — both rejected by the validator.
    guard_score() catches both cases with its if/else branches.

    Examples:
        safe_ratio(0, 5)  → 0.001  (if-branch: 0/5 == 0.0 → _SCORE_MIN)
        safe_ratio(5, 5)  → 0.999  (elif-branch: 5/5 == 1.0 → _SCORE_MAX)
        safe_ratio(2, 5)  → 0.4    (else-branch: interior, unchanged)
        safe_ratio(0, 0)  → 0.001  (if-branch: no tests ran → _SCORE_MIN)
    """
    if total <= 0:
        # No tests ran at all — return the floor, not zero.
        return _SCORE_MIN
    raw = passed / total          # may be exactly 0.0 or exactly 1.0
    return guard_score(raw)       # if/else catches both endpoints


def sanitize_any(obj: Any) -> Any:
    """Recursively walk *obj* and run guard_score() on every float found.

    The openenv validator recurses into every nested dict and list in the
    JSON response — reward.components, observation.metadata.weights, etc.
    This function is the last-resort safety net applied in env.py before
    any dict reaches Pydantic / FastAPI serialisation.
    """
    if isinstance(obj, dict):
        return {k: sanitize_any(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_any(v) for v in obj]
    if isinstance(obj, float):
        return guard_score(obj)   # if/else gate on every float in the tree
    # int, str, bool, None — not floats, pass through untouched
    return obj


def compute_destructive_penalty(files: Dict[str, str]) -> float:
    """Return a guarded (0, 1) penalty for destructive edits."""
    if not files:
        # No files at all — treat as maximally destructive, but still guard.
        return guard_score(0.3)

    empties = 0
    tiny = 0
    for content in files.values():
        stripped = (content or '').strip()
        if not stripped:
            empties += 1
        elif len(stripped) < 20:
            tiny += 1

    ratio = (empties + 0.5 * tiny) / max(1, len(files))
    raw_penalty = 0.3 * ratio
    # if raw_penalty is 0.0 (no empty files) guard_score maps it to _SCORE_MIN.
    # if raw_penalty is >= 1.0 (impossible here but defensive) maps to _SCORE_MAX.
    return guard_score(raw_penalty)


def compute_shaped_reward(
    tests_passed_ratio: float,
    improvement_over_last_step: float,
    steps_taken: int,
    destructive_action_penalty: float,
    w_pass: float,
    w_improve: float,
    w_step_penalty: float,
) -> float:
    """Generic shaped reward, always in the open interval (0, 1).

    Formula:
        reward = guard_score(tests_passed_ratio) * W_pass
               + guard_score(improvement_over_last_step) * W_improve
               - steps_taken * W_step_penalty
               - destructive_action_penalty

    Each input is guarded before use so that even a raw 0.0 or 1.0 passed
    in from a task is intercepted here by the if/else.  The final result is
    guarded again as a belt-and-suspenders last resort.
    """
    # Guard each input individually with the if/else before arithmetic.
    safe_pass   = guard_score(tests_passed_ratio)
    safe_improv = guard_score(improvement_over_last_step)

    score = (
        safe_pass   * w_pass
        + safe_improv * w_improve
        - steps_taken * w_step_penalty
        - destructive_action_penalty
    )
    # Final if/else guard on the result — catches any edge case in the formula.
    return guard_score(score)
