import pytest

from server.tasks.task_debug_hidden import DebugHiddenStateTask
from server.tasks.task_fix_api import FixBrokenApiTask
from server.tasks.task_resolve_ci import ResolveCIPipelineTask
from server.utils.graders import clamp_score, compute_destructive_penalty, compute_shaped_reward


def test_clamp_score_bounds():
    assert clamp_score(-1.0) == 0.001
    assert clamp_score(2.0) == 0.999
    assert clamp_score(0.42) == 0.42


def test_destructive_penalty_detects_empty_code():
    penalty = compute_destructive_penalty({"a.py": "", "b.py": "print('ok')"})
    assert 0.001 <= penalty <= 0.999


def test_shaped_reward_returns_bounded_score():
    reward = compute_shaped_reward(
        tests_passed_ratio=0.6,
        improvement_over_last_step=0.2,
        steps_taken=2,
        destructive_action_penalty=0.0,
        w_pass=1.0,
        w_improve=0.3,
        w_step_penalty=0.05,
    )
    assert 0.001 <= reward <= 0.999


def test_fix_api_grader_improves_after_submit():
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
    assert done


def test_resolve_ci_grader_reaches_full_pass():
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
    assert reward["tests_passed_ratio"] >= 0.99


def test_debug_hidden_submit_uses_hidden_tests():
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
    assert done
    assert reward["tests_passed_ratio"] >= 0.5
    assert info["hidden_total"] >= 1
