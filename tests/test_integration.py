"""
End-to-end integration test: simulates a full agent episode.
Run with:  pytest tests/test_integration.py -v
"""
import pytest
from openenv_software_dev.env import SoftwareDevEnv
from openenv_software_dev.actions import ActionType


def test_full_episode_bug_fix():
    """
    Simulate an agent that:
      1. Reads the broken file.
      2. Writes the fixed version.
      3. Runs the tests.
      4. Submits.
    Expected: episode terminates with done=True and positive reward.
    """
    env = SoftwareDevEnv(difficulty="easy", max_steps=20)
    obs, info = env.reset(seed=0)

    total_reward = 0.0
    done = trunc = False

    # Step 1: read the solution file
    obs, r, done, trunc, info = env.step({
        "type": ActionType.READ_FILE,
        "target_file": "solution.py",
        "text_input": "",
    })
    total_reward += r
    assert not done and not trunc

    # Step 2: write the fix
    obs, r, done, trunc, info = env.step({
        "type": ActionType.WRITE_FILE,
        "target_file": "solution.py",
        "text_input": "def add(a, b):\n    return a + b\n",
    })
    total_reward += r
    assert not done and not trunc

    # Step 3: run tests
    obs, r, done, trunc, info = env.step({
        "type": ActionType.RUN_TESTS,
        "target_file": "",
        "text_input": "",
    })
    total_reward += r

    # Step 4: submit
    obs, r, done, trunc, info = env.step({
        "type": ActionType.SUBMIT,
        "target_file": "",
        "text_input": "",
    })
    total_reward += r
    assert done is True


def test_episode_truncates_at_max_steps():
    """An agent that does nothing should get truncated after max_steps."""
    env = SoftwareDevEnv(difficulty="easy", max_steps=3)
    env.reset(seed=1)

    done = trunc = False
    steps = 0
    while not done and not trunc:
        _, _, done, trunc, _ = env.step({
            "type": ActionType.LIST_FILES,
            "target_file": "",
            "text_input": "",
        })
        steps += 1
        assert steps <= 5, "Episode should have ended by now"

    assert trunc is True or done is True


def test_gymnasium_spec_compliance():
    """Verify the environment conforms to the Gymnasium API contract."""
    import gymnasium as gym
    env = gym.make("SoftwareDev-v0")
    obs, info = env.reset()
    assert isinstance(obs, dict)
    obs, reward, done, trunc, info = env.step({
        "type": ActionType.SUBMIT,
        "target_file": "",
        "text_input": "",
    })
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()
