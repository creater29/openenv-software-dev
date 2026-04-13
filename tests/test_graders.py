"""
Unit tests for graders.py.

Every score assertion uses the open interval (0.001, 0.999) because the
openenv validator rejects any float exactly equal to 0.0 or 1.0 anywhere
in the response — including nested dicts and lists.
"""
from server.tasks.task_debug_hidden import DebugHiddenStateTask
from server.tasks.task_fix_api import FixBrokenApiTask
from server.tasks.task_resolve_ci import ResolveCIPipelineTask
from server.utils.graders import (
    _SCORE_MAX, _SCORE_MIN,
    clamp_score, compute_destructive_penalty,
    compute_shaped_reward, safe_ratio,
)


# ---------------------------------------------------------------------------
# clamp_score — must map any float into the open interval (_SCORE_MIN, _SCORE_MAX)
# ---------------------------------------------------------------------------

def test_clamp_score_low_values():
    """Values at or below 0.0 must clamp to _SCORE_MIN (0.001), never raw 0.0."""
    assert clamp_score(-1.0) == _SCORE_MIN
    assert clamp_score(0.0) == _SCORE_MIN


def test_clamp_score_high_values():
    """Values at or above 1.0 must clamp to _SCORE_MAX (0.999), never raw 1.0."""
    assert clamp_score(2.0) == _SCORE_MAX
    assert clamp_score(1.0) == _SCORE_MAX


def test_clamp_score_midrange():
    """Interior values should pass through unchanged."""
    assert clamp_score(0.42) == 0.42
    assert clamp_score(0.5) == 0.5


def test_clamp_score_strictly_open():
    """Output must be strictly inside (0, 1) for any input, no matter how extreme."""
    assert 0.0 < clamp_score(-99.0) < 1.0
    assert 0.0 < clamp_score(99.0) < 1.0


# ---------------------------------------------------------------------------
# safe_ratio — the single division site for passed/total conversions
# ---------------------------------------------------------------------------

def test_safe_ratio_all_pass():
    """3/3 must yield _SCORE_MAX (0.999), never raw 1.0."""
    result = safe_ratio(3, 3)
    assert result == _SCORE_MAX, f"Expected {_SCORE_MAX}, got {result}"
    assert result < 1.0


def test_safe_ratio_none_pass():
    """0/5 must yield _SCORE_MIN (0.001), never raw 0.0."""
    result = safe_ratio(0, 5)
    assert result == _SCORE_MIN, f"Expected {_SCORE_MIN}, got {result}"
    assert result > 0.0


def test_safe_ratio_zero_total():
    """0/0 (no tests ran) must yield _SCORE_MIN, never raw 0.0."""
    result = safe_ratio(0, 0)
    assert result == _SCORE_MIN


def test_safe_ratio_partial():
    """2/4 = 0.5 — an interior value — must pass through unchanged."""
    result = safe_ratio(2, 4)
    assert result == 0.5


def test_safe_ratio_always_open():
    """safe_ratio() must always produce a value strictly inside (0, 1)."""
    for passed, total in [(0, 0), (0, 5), (5, 5), (3, 5)]:
        r = safe_ratio(passed, total)
        assert 0.0 < r < 1.0, f"safe_ratio({passed},{total})={r} not in open interval"


# ---------------------------------------------------------------------------
# compute_destructive_penalty
# ---------------------------------------------------------------------------

def test_destructive_penalty_empty_file():
    penalty = compute_destructive_penalty({"a.py": "", "b.py": "print('ok')"})
    assert 0.0 < penalty < 1.0


def test_destructive_penalty_healthy_files():
    penalty = compute_destructive_penalty({"a.py": "def foo(): pass\n" * 5})
    assert _SCORE_MIN <= penalty <= _SCORE_MAX


def test_destructive_penalty_no_files():
    """Empty file dict must return a safe non-zero value."""
    penalty = compute_destructive_penalty({})
    assert 0.0 < penalty < 1.0


# ---------------------------------------------------------------------------
# compute_shaped_reward
# ---------------------------------------------------------------------------

def test_shaped_reward_normal_case():
    reward = compute_shaped_reward(0.6, 0.2, 2, 0.0, 0.75, 0.25, 0.03)
    assert 0.0 < reward < 1.0


def test_shaped_reward_worst_case():
    """Even with zero tests passed and extreme penalty, reward stays > 0."""
    reward = compute_shaped_reward(0.0, 0.0, 100, 0.5, 0.75, 0.25, 0.03)
    assert 0.0 < reward < 1.0


def test_shaped_reward_best_case():
    """Even with perfect scores and no penalty, reward stays < 1."""
    reward = compute_shaped_reward(1.0, 1.0, 0, 0.0, 0.75, 0.25, 0.0)
    assert 0.0 < reward < 1.0


# ---------------------------------------------------------------------------
# Task-level integration — all numeric fields must be in open interval (0, 1)
# ---------------------------------------------------------------------------

def _all_floats_open(obj, path=""):
    """Recursively assert every float in obj is in (0.0, 1.0) exclusive."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _all_floats_open(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _all_floats_open(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        assert 0.0 < obj < 1.0, f"Float at {path}={obj} is NOT in open interval (0,1)"


def test_fix_api_correct_submit():
    """Submitting the correct fix must raise tests_passed_ratio and set done=True."""
    task = FixBrokenApiTask()
    task.reset("easy")
    initial = task.last_pass_ratio
    fixed_code = (
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
        "def compute_total(items):\n    return sum(items)\n\n"
        "@app.get('/total')\ndef total_route():\n"
        "    values = [1, 2, 3]\n    return {'total': compute_total(values)}\n"
    )
    reward, done, _ = task.step({"action": "submit", "code": fixed_code})
    assert reward["tests_passed_ratio"] >= initial
    _all_floats_open(reward)
    assert done


def test_fix_api_inspect_all_fields_open():
    """Every numeric field in the reward dict must be strictly inside (0, 1)."""
    task = FixBrokenApiTask()
    task.reset("medium")
    reward, _, _ = task.step({"action": "inspect"})
    _all_floats_open(reward)


def test_resolve_ci_full_pass_reaches_score_max():
    """After correct patch + run_tests, tests_passed_ratio must be _SCORE_MAX (0.999)."""
    task = ResolveCIPipelineTask()
    task.reset("medium")
    patched_utils = (
        "def normalize(values):\n    if not values:\n        return []\n"
        "    total = sum(values)\n    return [v / total for v in values]\n\n"
        "def moving_average(values):\n    if len(values) < 2:\n        return values\n"
        "    out = []\n    for i in range(len(values) - 1):\n"
        "        out.append((values[i] + values[i + 1]) / 2)\n    return out\n"
    )
    task.step({"action": "patch", "filename": "utils.py", "code": patched_utils})
    reward, _, _ = task.step({"action": "run_tests"})
    assert reward["tests_passed_ratio"] == _SCORE_MAX, (
        f"All tests should yield _SCORE_MAX={_SCORE_MAX}, got {reward['tests_passed_ratio']}"
    )
    _all_floats_open(reward)


def test_resolve_ci_all_reward_fields_open():
    task = ResolveCIPipelineTask()
    task.reset("medium")
    reward, _, _ = task.step({"action": "run_tests"})
    _all_floats_open(reward)


def test_debug_hidden_submit_all_fields_open():
    """submit must run hidden tests, set done=True, and keep all floats in (0,1)."""
    task = DebugHiddenStateTask()
    task.reset("hard")
    fixed_config = (
        "DEFAULT_CONFIG = {'threshold': 10, 'window': 2}\n"
        "RUNTIME_CONFIG = dict(DEFAULT_CONFIG)\n\n"
        "def get_config():\n    return RUNTIME_CONFIG\n\n"
        "def reset_runtime():\n    global RUNTIME_CONFIG\n"
        "    RUNTIME_CONFIG = dict(DEFAULT_CONFIG)\n"
    )
    task.step({"action": "patch", "filename": "config.py", "code": fixed_config})
    reward, done, info = task.step({"action": "submit"})
    assert done is True
    assert reward["tests_passed_ratio"] >= _SCORE_MIN
    _all_floats_open(reward)
    assert int(info["hidden_total"]) >= 1


def test_debug_hidden_safe_ratio_in_run_tests():
    """run_tests must never return a raw 0.0 or 1.0 in visible_ratio."""
    task = DebugHiddenStateTask()
    task.reset("hard")
    # Don't patch anything — visible tests should fail → ratio is _SCORE_MIN
    reward, _, info = task.step({"action": "run_tests"})
    _all_floats_open(reward)
    assert 0.0 < task.visible_ratio < 1.0
