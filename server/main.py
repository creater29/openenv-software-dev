from __future__ import annotations

import random
from fastapi import FastAPI, HTTPException

from server.env import ActionModel, OpenEnvSWEEnv, ResetRequest, StateResponse, StepResponse
from server.utils.graders import sanitize_any

app = FastAPI(title="SWE-Sim OpenEnv Server", version="1.0.0")
env = OpenEnvSWEEnv()

_TASKS = ["fix_broken_api", "resolve_ci_pipeline", "debug_hidden_state"]
_DIFFICULTIES = ["easy", "medium", "hard"]

# Sentinel reward returned by /reset.
# Every float must be strictly inside (0, 1) — the openenv validator rejects
# exactly 0.0 and exactly 1.0 in ANY numeric field of the response.
# sanitize_any() is called at module load so that any future edit to these
# literal values is automatically clamped before the server ever starts.
_RESET_REWARD = sanitize_any({
    "reward": 0.001,
    "tests_passed_ratio": 0.001,
    "improvement_over_last_step": 0.001,
    "step_penalty": 0.001,
    "destructive_action_penalty": 0.001,
    "components": {"reset": 0.001},
})


@app.get("/")
def root() -> dict:
    return {"name": "SWE-Sim", "status": "ok"}


@app.post("/reset", response_model=StepResponse)
def reset(request: ResetRequest = ResetRequest()) -> StepResponse:
    """Reset the environment to a clean episode.

    If the validator sends an empty body, sensible defaults are chosen.
    """
    task = request.task or random.choice(_TASKS)
    difficulty = request.difficulty or "medium"

    try:
        obs = env.reset(task=task, difficulty=difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StepResponse(
        observation=obs,
        reward=_RESET_REWARD,
        done=False,
        info={"message": "Environment reset", "task": task, "difficulty": difficulty},
    )


@app.post("/step", response_model=StepResponse)
def step(action: ActionModel) -> StepResponse:
    """Advance the episode by one step."""
    try:
        observation, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    """Return the current environment state snapshot."""
    return env.state()
