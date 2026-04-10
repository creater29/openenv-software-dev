from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from server.tasks.task_debug_hidden import DebugHiddenStateTask
from server.tasks.task_fix_api import FixBrokenApiTask
from server.tasks.task_resolve_ci import ResolveCIPipelineTask
from server.utils.graders import sanitize_any


class ObservationModel(BaseModel):
    task: str = Field(description="Current task identifier.")
    difficulty: str = Field(description="Current difficulty level.")
    step: int = Field(description="Current step count.")
    max_steps: int = Field(description="Maximum allowed steps.")
    files: Optional[Dict[str, str]] = Field(default=None)
    file: Optional[str] = Field(default=None)
    code: Optional[str] = Field(default=None)
    error_log: Optional[str] = Field(default=None)
    test_results: Optional[str] = Field(default=None)
    visible_tests: Optional[str] = Field(default=None)
    hidden_tests: Optional[str] = Field(default=None)
    last_reward: float = Field(description="Reward from previous step.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionModel(BaseModel):
    action: str
    filename: Optional[str] = Field(default=None)
    code: Optional[str] = Field(default=None)


class RewardModel(BaseModel):
    reward: float
    tests_passed_ratio: float
    improvement_over_last_step: float
    step_penalty: float
    destructive_action_penalty: float
    components: Dict[str, float] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task: Optional[str] = Field(default=None)
    difficulty: Optional[str] = Field(default=None)


class StepResponse(BaseModel):
    observation: ObservationModel
    reward: RewardModel
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    ready: bool
    task: Optional[str] = None
    difficulty: Optional[str] = None
    step: int = 0
    max_steps: int = 0
    done: bool = False
    score: float = 0.001
    internal: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class _EpisodeState:
    task: Any
    done: bool = False


def _build_observation(task_obj: Any, last_reward: float) -> ObservationModel:
    """Build ObservationModel, sanitizing ALL floats in the observation dict."""
    raw = task_obj.observation(last_reward=last_reward)
    clean = sanitize_any(raw)
    return ObservationModel(**clean)


def _build_reward(reward_dict: Dict[str, Any]) -> RewardModel:
    """Build RewardModel, sanitizing ALL floats including nested components."""
    clean = sanitize_any(reward_dict)
    return RewardModel(**clean)


class OpenEnvSWEEnv:
    def __init__(self) -> None:
        self._episode: Optional[_EpisodeState] = None
        self._task_registry = {
            "fix_broken_api": FixBrokenApiTask,
            "resolve_ci_pipeline": ResolveCIPipelineTask,
            "debug_hidden_state": DebugHiddenStateTask,
        }

    def reset(self, task: str, difficulty: str = "medium") -> ObservationModel:
        if task not in self._task_registry:
            raise ValueError(f"Unknown task: {task}")
        task_obj = self._task_registry[task]()
        task_obj.reset(difficulty=difficulty)
        self._episode = _EpisodeState(task=task_obj, done=False)
        # Use 0.001 as initial last_reward — 0.0 is rejected by the validator
        return _build_observation(task_obj, last_reward=0.001)

    def step(self, action: ActionModel) -> Tuple[ObservationModel, RewardModel, bool, Dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._episode.done:
            raise RuntimeError("Episode already finished. Call reset() for a new episode.")

        reward_dict, done, info = self._episode.task.step(action.model_dump())
        self._episode.done = done

        observation = _build_observation(self._episode.task, last_reward=reward_dict["reward"])
        reward = _build_reward(reward_dict)
        return observation, reward, done, sanitize_any(info)

    def state(self) -> StateResponse:
        if self._episode is None:
            return StateResponse(ready=False)
        task = self._episode.task
        return StateResponse(
            ready=True,
            task=task.name,
            difficulty=task.difficulty,
            step=task.steps_taken,
            max_steps=task.max_steps,
            done=self._episode.done,
            score=max(0.001, min(0.999, task.current_score)),
            internal=sanitize_any(task.state()),
        )
