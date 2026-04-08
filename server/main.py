from fastapi import FastAPI, HTTPException

import random
from server.env import ActionModel, OpenEnvSWEEnv, ResetRequest, StateResponse, StepResponse

app = FastAPI(title="SWE-Sim OpenEnv Server", version="1.0.0")
env = OpenEnvSWEEnv()

_TASKS = ["fix_broken_api", "resolve_ci_pipeline", "debug_hidden_state"]
_DIFFICULTIES = ["easy", "medium", "hard"]


@app.get("/")
def root() -> dict:
    return {"name": "SWE-Sim", "status": "ok"}


@app.post("/reset", response_model=StepResponse)
def reset(request: ResetRequest = ResetRequest()) -> StepResponse:
    # If the validator sends an empty body, pick sensible defaults
    task = request.task or random.choice(_TASKS)
    difficulty = request.difficulty or "medium"
    try:
        obs = env.reset(task=task, difficulty=difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StepResponse(
        observation=obs,
        reward={
            "reward": 0.0,
            "tests_passed_ratio": 0.0,
            "improvement_over_last_step": 0.0,
            "step_penalty": 0.0,
            "destructive_action_penalty": 0.0,
            "components": {"reset": 0.0},
        },
        done=False,
        info={"message": "Environment reset", "task": task, "difficulty": difficulty},
    )


@app.post("/step", response_model=StepResponse)
def step(action: ActionModel) -> StepResponse:
    try:
        observation, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return env.state()
