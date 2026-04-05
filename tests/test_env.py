"""
Tests for the core SoftwareDevEnv gymnasium environment.
Run with:  pytest tests/test_env.py -v
"""
import pytest
from openenv_software_dev.env import SoftwareDevEnv
from openenv_software_dev.actions import ActionType


@pytest.fixture
def env():
    e = SoftwareDevEnv(difficulty="easy", max_steps=10)
    yield e


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert "task_id" in obs
    assert "files" in obs
    assert "task_description" in info


def test_reset_populates_filesystem(env):
    obs, _ = env.reset(seed=42)
    assert len(obs["file_tree"]) > 0, "Starter files should be present after reset"


def test_read_file_action(env):
    env.reset(seed=42)
    obs, reward, done, trunc, info = env.step({
        "type": ActionType.READ_FILE,
        "target_file": "solution.py",
        "text_input": "",
    })
    assert info["action_result"]["status"] == "success"
    assert "content" in info["action_result"]


def test_write_file_action(env):
    env.reset(seed=42)
    obs, reward, done, trunc, info = env.step({
        "type": ActionType.WRITE_FILE,
        "target_file": "solution.py",
        "text_input": "def add(a, b):\n    return a + b\n",
    })
    assert info["action_result"]["status"] == "success"
    assert obs["files"]["solution.py"] == "def add(a, b):\n    return a + b\n"


def test_run_tests_action(env):
    env.reset(seed=42)
    obs, reward, done, trunc, info = env.step({
        "type": ActionType.RUN_TESTS,
        "target_file": "",
        "text_input": "",
    })
    assert info["action_result"]["status"] in ("success", "failure", "timeout", "error")


def test_submit_terminates_episode(env):
    env.reset(seed=42)
    _, _, done, trunc, _ = env.step({
        "type": ActionType.SUBMIT,
        "target_file": "",
        "text_input": "",
    })
    assert done is True


def test_max_steps_truncates_episode(env):
    env.reset(seed=42)
    action = {"type": ActionType.LIST_FILES, "target_file": "", "text_input": ""}
    done = trunc = False
    for _ in range(10):
        _, _, done, trunc, _ = env.step(action)
        if done or trunc:
            break
    assert trunc is True or done is True


def test_reward_is_float(env):
    env.reset(seed=42)
    _, reward, _, _, _ = env.step({
        "type": ActionType.RUN_TESTS,
        "target_file": "",
        "text_input": "",
    })
    assert isinstance(reward, float)


def test_list_files_action(env):
    env.reset(seed=42)
    obs, _, _, _, info = env.step({
        "type": ActionType.LIST_FILES,
        "target_file": "",
        "text_input": "",
    })
    result = info["action_result"]
    assert result["status"] == "success"
    assert isinstance(result["files"], list)
