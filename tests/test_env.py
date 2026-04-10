from server.env import ActionModel, OpenEnvSWEEnv


def test_reset_returns_clean_reproducible_state():
    env = OpenEnvSWEEnv()
    obs1 = env.reset(task="fix_broken_api", difficulty="easy")
    code1 = obs1.code

    env.step(ActionModel(action="inspect"))
    obs2 = env.reset(task="fix_broken_api", difficulty="easy")
    code2 = obs2.code

    assert code1 == code2
    assert obs2.step == 0


def test_state_endpoint_payload_shape():
    env = OpenEnvSWEEnv()
    env.reset(task="resolve_ci_pipeline", difficulty="medium")
    state = env.state()
    assert state.ready is True
    assert state.task == "resolve_ci_pipeline"
    assert 0.001 <= state.score <= 0.999


def test_step_score_bounds_for_all_tasks():
    env = OpenEnvSWEEnv()

    env.reset(task="fix_broken_api", difficulty="easy")
    _, reward_1, _, _ = env.step(ActionModel(action="inspect"))

    env.reset(task="resolve_ci_pipeline", difficulty="medium")
    _, reward_2, _, _ = env.step(ActionModel(action="run_tests"))

    env.reset(task="debug_hidden_state", difficulty="hard")
    _, reward_3, _, _ = env.step(ActionModel(action="run_tests"))

    assert 0.001 <= reward_1.reward <= 0.999
    assert 0.001 <= reward_2.reward <= 0.999
    assert 0.001 <= reward_3.reward <= 0.999


def test_episode_ends_on_max_steps():
    env = OpenEnvSWEEnv()
    obs = env.reset(task="fix_broken_api", difficulty="easy")
    done = False
    for _ in range(obs.max_steps):
        _, _, done, _ = env.step(ActionModel(action="inspect"))
    assert done is True
