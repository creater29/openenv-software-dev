"""
Unit tests for graders.py.

Every score assertion uses the open interval (0.001, 0.999).  The tests
explicitly exercise the if/else branches of guard_score() to prove that
neither 0.0 nor 1.0 can survive into any response field.
"""
from server.tasks.task_debug_hidden import DebugHiddenStateTask
from server.tasks.task_fix_api import FixBrokenApiTask
from server.tasks.task_resolve_ci import ResolveCIPipelineTask
from server.utils.graders import (
    _SCORE_MAX, _SCORE_MIN,
    guard_score, compute_destructive_penalty,
    compute_shaped_reward, safe_ratio,
)


# ---------------------------------------------------------------------------
# guard_score — the explicit if/else gate
# ---------------------------------------------------------------------------

def test_guard_score_if_branch_zero():
    """if score == 0 → return 0.001  (the if-branch)"""
    assert guard_score(0.0) == _SCORE_MIN       # exact zero → floor
    assert guard_score(-5.0) == _SCORE_MIN      # negative → floor
    assert guard_score(-0.0) == _SCORE_MIN      # negative zero → floor

def test_guard_score_elif_branch_one():
    """elif score == 1 → return 0.999  (the elif-branch)"""
    assert guard_score(1.0) == _SCORE_MAX       # exact one → ceiling
    assert guard_score(2.5) == _SCORE_MAX       # above one → ceiling
    assert guard_score(99.0) == _SCORE_MAX      # far above one → ceiling

def test_guard_score_else_branch_interior():
    """else → return score unchanged  (the else-branch)"""
    assert guard_score(0.42) == 0.42            # interior value passes through
    assert guard_score(0.5)  == 0.5
    assert guard_score(0.001) == 0.001          # exactly at floor still passes
    assert guard_score(0.999) == 0.999          # exactly at ceiling still passes

def test_guard_score_never_returns_endpoint():
    """guard_score must NEVER return exactly 0.0 or 1.0 for any input."""
    for v in [-99, -1, 0, 0.0, 1.0, 1, 2, 99]:
        result = guard_score(v)
        assert result != 0.0, f"guard_score({v}) returned 0.0"
        assert result != 1.0, f"guard_score({v}) returned 1.0"
        assert 0.0 < result < 1.0, f"guard_score({v})={result} not in (0,1)"


# ---------------------------------------------------------------------------
# safe_ratio — single division site, runs guard_score internally
# ---------------------------------------------------------------------------

def test_safe_ratio_all_pass_hits_elif_branch():
    """3/3 = 1.0 → guard_score elif-branch → _SCORE_MAX (0.999), never 1.0."""
    result = safe_ratio(3, 3)
    assert result == _SCORE_MAX
    assert result != 1.0

def test_safe_ratio_none_pass_hits_if_branch():
    """0/5 = 0.0 → guard_score if-branch → _SCORE_MIN (0.001), never 0.0."""
    result = safe_ratio(0, 5)
    assert result == _SCORE_MIN
    assert result != 0.0

def test_safe_ratio_zero_total_hits_if_branch():
    """0/0 = no tests → _SCORE_MIN directly, never 0.0."""
    result = safe_ratio(0, 0)
    assert result == _SCORE_MIN
    assert result != 0.0

def test_safe_ratio_partial_hits_else_branch():
    """2/4 = 0.5 → else-branch → 0.5 unchanged."""
    assert safe_ratio(2, 4) == 0.5

def test_safe_ratio_always_strictly_open():
    """safe_ratio must always produce a value strictly inside (0, 1)."""
    for passed, total in [(0, 0), (0, 5), (5, 5), (2, 5), (1, 1)]:
        r = safe_ratio(passed, total)
        assert 0.0 < r < 1.0, f"safe_ratio({passed},{total})={r} not in (0,1)"


# ---------------------------------------------------------------------------
# compute_destructive_penalty
# ---------------------------------------------------------------------------

def test_destructive_penalty_empty_file_non_zero():
    penalty = compute_destructive_penalty({"a.py": "", "b.py": "print('ok')"})
    assert 0.0 < penalty < 1.0

def test_destructive_penalty_healthy_files_at_floor():
    penalty = compute_destructive_penalty({"a.py": "def foo(): pass\n" * 5})
    assert _SCORE_MIN <= penalty <= _SCORE_MAX

def test_destructive_penalty_no_files_non_zero():
    penalty = compute_destructive_penalty({})
    assert 0.0 < penalty < 1.0


# ---------------------------------------------------------------------------
# compute_shaped_reward
# ---------------------------------------------------------------------------

def test_shaped_reward_normal_open_interval():
    r = compute_shaped_reward(0.6, 0.2, 2, 0.0, 0.75, 0.25, 0.03)
    assert 0.0 < r < 1.0

def test_shaped_reward_worst_stays_above_zero():
    r = compute_shaped_reward(0.0, 0.0, 100, 0.5, 0.75, 0.25, 0.03)
    assert 0.0 < r < 1.0

def test_shaped_reward_best_stays_below_one():
    r = compute_shaped_reward(1.0, 1.0, 0, 0.0, 0.75, 0.25, 0.0)
    assert 0.0 < r < 1.0


# ---------------------------------------------------------------------------
# Recursive float checker used by integration tests
# ---------------------------------------------------------------------------

def _check_all_floats(obj, path=""):
    """Walk obj recursively; assert every float is strictly in (0, 1)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_all_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_all_floats(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        assert 0.0 < obj < 1.0, (
            f"Float at {path}={obj!r} is NOT in the open interval (0, 1)."
            f"  Expected: 0.001 for zero, 0.999 for one."
        )


# ---------------------------------------------------------------------------
# Task-level integration — all reward dict floats must be in (0, 1)
# ---------------------------------------------------------------------------

def test_fix_api_correct_submit_all_floats_open():
    """Submitting the correct fix: done=True and every float in reward is open."""
    task = FixBrokenApiTask()
    task.reset("easy")
    initial = task.last_pass_ratio
    fixed = (
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
        "def compute_total(items):\n    return sum(items)\n\n"
        "@app.get('/total')\ndef total_route():\n"
        "    values = [1, 2, 3]\n    return {'total': compute_total(values)}\n"
    )
    reward, done, _ = task.step({"action": "submit", "code": fixed})
    assert reward["tests_passed_ratio"] >= initial
    _check_all_floats(reward)
    assert done

def test_fix_api_inspect_all_floats_open():
    task = FixBrokenApiTask()
    task.reset("medium")
    reward, _, _ = task.step({"action": "inspect"})
    _check_all_floats(reward)

def test_resolve_ci_full_pass_is_score_max():
    """After correct patch, tests_passed_ratio == _SCORE_MAX (0.999), never 1.0."""
    task = ResolveCIPipelineTask()
    task.reset("medium")
    patched = (
        "def normalize(values):\n    if not values:\n        return []\n"
        "    total = sum(values)\n    return [v / total for v in values]\n\n"
        "def moving_average(values):\n    if len(values) < 2:\n        return values\n"
        "    out = []\n    for i in range(len(values) - 1):\n"
        "        out.append((values[i] + values[i + 1]) / 2)\n    return out\n"
    )
    task.step({"action": "patch", "filename": "utils.py", "code": patched})
    reward, _, _ = task.step({"action": "run_tests"})
    # elif-branch of guard_score fires: 5/5=1.0 → 0.999
    assert reward["tests_passed_ratio"] == _SCORE_MAX
    assert reward["tests_passed_ratio"] != 1.0
    _check_all_floats(reward)

def test_resolve_ci_run_tests_all_floats_open():
    task = ResolveCIPipelineTask()
    task.reset("medium")
    reward, _, _ = task.step({"action": "run_tests"})
    _check_all_floats(reward)

def test_debug_hidden_submit_all_floats_open():
    """submit: done=True, hidden tests evaluated, all floats in (0,1)."""
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
    _check_all_floats(reward)
    assert int(info["hidden_total"]) >= 1

def test_debug_hidden_safe_ratio_never_zero_or_one():
    """run_tests on unpatched code: visible_ratio must be in (0,1), never 0.0."""
    task = DebugHiddenStateTask()
    task.reset("hard")
    reward, _, _ = task.step({"action": "run_tests"})
    _check_all_floats(reward)
    # Even with 0 tests passing, safe_ratio hits the if-branch → 0.001
    assert 0.0 < task.visible_ratio < 1.0
    assert task.visible_ratio != 0.0
