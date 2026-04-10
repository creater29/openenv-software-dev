"""
Integration tests for the OpenEnvSWEEnv facade.

All score/reward assertions use the open interval (0.001, 0.999) because the
openenv validator rejects any float == 0.0 or == 1.0 anywhere in the response.
"""
from server.env import ActionModel, OpenEnvSWEEnv
from server.utils.graders import _SCORE_MAX, _SCORE_MIN


def test_reset_returns_clean_reproducible_state():
    """reset() must produce an identical observation every time (no leaked state)."""
    env = OpenEnvSWEEnv()
    obs1 = env.reset(task="fix_broken_api", difficulty="easy")
    code1 = obs1.code

    env.step(ActionModel(action="inspect"))          # dirty the episode
    obs2 = env.reset(task="fix_broken_api", difficulty="easy")
    code2 = obs2.code

    assert code1 == code2, "reset() must return identical code regardless of prior steps"
    assert obs2.step == 0, "reset() must zero the step counter"


def test_initial_observation_floats_in_open_interval():
    """Every float in the initial observation must be strictly inside (0, 1)."""
    env = OpenEnvSWEEnv()
    obs = env.reset(task="fix_broken_api", difficulty="easy")
    assert 0.0 < obs.last_reward < 1.0, f"last_reward={obs.last_reward} not in (0,1)"


def test_state_endpoint_payload_shape():
    """GET /state must return a well-formed StateResponse with open-interval score."""
    env = OpenEnvSWEEnv()
    env.reset(task="resolve_ci_pipeline", difficulty="medium")
    state = env.state()
    assert state.ready is True
    assert state.task == "resolve_ci_pipeline"
    # Score must be strictly inside (0, 1) — never 0.0 or 1.0
    assert _SCORE_MIN <= state.score <= _SCORE_MAX, f"score={state.score} out of open interval"


def test_step_score_bounds_for_all_tasks():
    """reward.reward must be strictly inside (0, 1) for every task."""
    env = OpenEnvSWEEnv()

    env.reset(task="fix_broken_api", difficulty="easy")
    _, reward_1, _, _ = env.step(ActionModel(action="inspect"))

    env.reset(task="resolve_ci_pipeline", difficulty="medium")
    _, reward_2, _, _ = env.step(ActionModel(action="run_tests"))

    env.reset(task="debug_hidden_state", difficulty="hard")
    _, reward_3, _, _ = env.step(ActionModel(action="run_tests"))

    for tag, r in [("fix_broken_api", reward_1), ("resolve_ci", reward_2), ("debug_hidden", reward_3)]:
        assert 0.0 < r.reward < 1.0, f"{tag}: reward={r.reward} not in open interval (0,1)"
        assert 0.0 < r.tests_passed_ratio < 1.0, f"{tag}: tests_passed_ratio={r.tests_passed_ratio} not in open interval"
        assert 0.0 < r.step_penalty < 1.0, f"{tag}: step_penalty={r.step_penalty} not in open interval"
        assert 0.0 < r.destructive_action_penalty < 1.0, f"{tag}: destructive_penalty={r.destructive_action_penalty} not in open interval"


def test_reward_components_in_open_interval():
    """All named reward components must be strictly inside (0, 1)."""
    env = OpenEnvSWEEnv()
    env.reset(task="resolve_ci_pipeline", difficulty="medium")
    _, reward, _, _ = env.step(ActionModel(action="run_tests"))
    for key, val in reward.components.items():
        assert 0.0 < val < 1.0, f"component[{key}]={val} not in open interval (0,1)"


def test_episode_ends_on_max_steps():
    """Episode must set done=True when max_steps is exhausted."""
    env = OpenEnvSWEEnv()
    obs = env.reset(task="fix_broken_api", difficulty="easy")
    done = False
    for _ in range(obs.max_steps):
        _, _, done, _ = env.step(ActionModel(action="inspect"))
    assert done is True, "Episode must be done after max_steps"


def test_reset_after_done_works():
    """reset() must succeed even if the episode is already done."""
    env = OpenEnvSWEEnv()
    obs = env.reset(task="fix_broken_api", difficulty="easy")
    for _ in range(obs.max_steps):
        env.step(ActionModel(action="inspect"))
    # Must not raise
    obs2 = env.reset(task="fix_broken_api", difficulty="easy")
    assert obs2.step == 0
