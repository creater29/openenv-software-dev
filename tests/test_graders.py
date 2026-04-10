"""
Unit tests for graders.py.

All score assertions use the open-interval bounds (0.001, 0.999) because the
openenv validator rejects any float == 0.0 or == 1.0 anywhere in the response.
"""
from server.tasks.task_debug_hidden import DebugHiddenStateTask
from server.tasks.task_fix_api import FixBrokenApiTask
from server.tasks.task_resolve_ci import ResolveCIPipelineTask
from server.utils.graders import (
    _SCORE_MAX,
    _SCORE_MIN,
    clamp_score,
    compute_destructive_penalty,
    compute_shaped_reward,
)


# ---------------------------------------------------------------------------
# clamp_score
# ---------------------------------------------------------------------------

def test_clamp_score_low_value():
    """Values <= 0.0 must clamp to _SCORE_MIN (0.001), never raw 0.0."""
    assert clamp_score(-1.0) == _SCORE_MIN
    assert clamp_score(0.0) == _SCORE_MIN


def test_clamp_score_high_value():
    """Values >= 1.0 must clamp to _SCORE_MAX (0.999), never raw 1.0."""
    assert clamp_score(2.0) == _SCORE_MAX
    assert clamp_score(1.0) == _SCORE_MAX


def test_clamp_score_midrange():
    """Values in the interior pass through unchanged."""
    assert clamp_score(0.42) == 0.42
    assert clamp_score(0.5) == 0.5


def test_clamp_score_never_hits_endpoints():
    """Confirm output is strictly inside (0, 1) for extreme inputs."""
    assert 0.0 < clamp_score(-99.0) < 1.0
    assert 0.0 < clamp_score(99.0) < 1.0


# ---------------------------------------------------------------------------
# compute_destructive_penalty
# ---------------------------------------------------------------------------

def test_destructive_penalty_empty_file():
    """Empty file must produce a non-zero, sub-1 penalty."""
    penalty = compute_destructive_penalty({"a.py": "", "b.py": "print('ok')"})
    assert 0.0 < penalty < 1.0


def test_destructive_penalty_all_healthy():
    """Normal-size files should yield the minimum penalty (_SCORE_MIN)."""
    penalty = compute_destructive_penalty({"a.py": "def foo(): pass\n" * 5})
    assert _SCORE_MIN <= penalty <= _SCORE_MAX


def test_destructive_penalty_no_files():
    """Empty file dict must still return a safe non-zero value."""
    penalty = compute_destructive_penalty({})
    assert 0.0 < penalty < 1.0


# ---------------------------------------------------------------------------
# compute_shaped_reward
# ---------------------------------------------------------------------------

def test_shaped_reward_open_interval():
    """Shaped reward must be strictly inside (0, 1) for normal inputs."""
    reward = compute_shaped_reward(0.6, 0.2, 2, 0.0, 1.0, 0.3, 0.05)
    assert 0.0 < reward < 1.0


def test_shaped_reward_worst_case():
    """Even with zero tests passed and max penalty, reward stays > 0."""
    reward = compute_shaped_reward(0.0, 0.0, 100, 0.5, 1.0, 0.3, 0.05)
    assert 0.0 < reward < 1.0


def test_shaped_reward_best_case():
    """Even with perfect scores and no penalty, reward stays < 1."""
    reward = compute_shaped_reward(1.0, 1.0, 0, 0.0, 1.0, 0.3, 0.0)
    assert 0.0 < reward < 1.0


# ---------------------------------------------------------------------------
# Task-level grader integration
# ---------------------------------------------------------------------------

def test_fix_api_grader_improves_after_correct_submit():
    """Submitting the correct fix must increase tests_passed_ratio."""
    task = FixBrokenApiTask()
    task.reset("easy")
    initial = task.last_pass_ratio
    fixed_code = (
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "def compute_total(items):\n"
        "    return sum(items)\n\n"
        "@app.get('/total')\n"
        "def total_route():\n"
        "    values = [1, 2, 3]\n"
        "    return {'total': compute_total(values)}\n"
    )
    reward, done, _ = task.step({"action": "submit", "code": fixed_code})
    assert reward["tests_passed_ratio"] >= initial
    assert 0.0 < reward["reward"] < 1.0
    assert done


def test_fix_api_reward_all_fields_open_interval():
    """Every numeric field in the reward dict must be in the open interval."""
    task = FixBrokenApiTask()
    task.reset("medium")
    reward, _, _ = task.step({"action": "inspect"})
    for key in ("reward", "tests_passed_ratio", "step_penalty", "destructive_action_penalty"):
        val = reward[key]
        assert 0.0 < val < 1.0, f"reward[{key}]={val} not in open interval (0,1)"
    for k, v in reward["components"].items():
        assert 0.0 < v < 1.0, f"component[{k}]={v} not in open interval (0,1)"


def test_resolve_ci_grader_reaches_score_max():
    """After correct patch + run_tests, tests_passed_ratio == _SCORE_MAX (0.999)."""
    task = ResolveCIPipelineTask()
    task.reset("medium")
    patched_utils = (
        "def normalize(values):\n"
        "    if not values:\n"
        "        return []\n"
        "    total = sum(values)\n"
        "    return [v / total for v in values]\n\n"
        "def moving_average(values):\n"
        "    if len(values) < 2:\n"
        "        return values\n"
        "    out = []\n"
        "    for i in range(len(values) - 1):\n"
        "        out.append((values[i] + values[i + 1]) / 2)\n"
        "    return out\n"
    )
    task.step({"action": "patch", "filename": "utils.py", "code": patched_utils})
    reward, _, _ = task.step({"action": "run_tests"})
    # Clamped to _SCORE_MAX (0.999) — never raw 1.0
    assert reward["tests_passed_ratio"] == _SCORE_MAX, (
        f"Expected _SCORE_MAX={_SCORE_MAX}, got {reward['tests_passed_ratio']}"
    )
    assert 0.0 < reward["reward"] < 1.0


def test_debug_hidden_submit_evaluates_hidden_tests():
    """submit action must run hidden tests and set done=True."""
    task = DebugHiddenStateTask()
    task.reset("hard")
    fixed_config = (
        "DEFAULT_CONFIG = {'threshold': 10, 'window': 2}\n"
        "RUNTIME_CONFIG = dict(DEFAULT_CONFIG)\n\n"
        "def get_config():\n"
        "    return RUNTIME_CONFIG\n\n"
        "def reset_runtime():\n"
        "    global RUNTIME_CONFIG\n"
        "    RUNTIME_CONFIG = dict(DEFAULT_CONFIG)\n"
    )
    task.step({"action": "patch", "filename": "config.py", "code": fixed_config})
    reward, done, info = task.step({"action": "submit"})
    assert done is True
    assert reward["tests_passed_ratio"] >= _SCORE_MIN
    assert 0.0 < reward["reward"] < 1.0
    assert info["hidden_total"] >= 1
