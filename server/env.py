from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from server.tasks.task_debug_hidden import DebugHiddenStateTask
from server.tasks.task_fix_api import FixBrokenApiTask
from server.tasks.task_resolve_ci import ResolveCIPipelineTask


class ObservationModel(BaseModel):
    """Typed environment observation returned at every transition."""

    task: str = Field(description="Current task identifier.")
    difficulty: str = Field(description="Current difficulty level: easy, medium, or hard.")
    step: int = Field(description="Current step count in this episode.")
    max_steps: int = Field(description="Maximum allowed steps for this episode.")
    files: Optional[Dict[str, str]] = Field(default=None, description="Map of filename to file contents for multi-file tasks.")
    file: Optional[str] = Field(default=None, description="Primary file name for single-file tasks.")
    code: Optional[str] = Field(default=None, description="Primary code blob for single-file tasks.")
    error_log: Optional[str] = Field(default=None, description="Latest traceback or test failure output.")
    test_results: Optional[str] = Field(default=None, description="Human-readable summary of current test status.")
    visible_tests: Optional[str] = Field(default=None, description="Visible test status for hidden-state tasks.")
    hidden_tests: Optional[str] = Field(default=None, description="Hidden test status description.")
    last_reward: float = Field(description="Reward returned by the previous step.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Task-specific metadata useful for agent reasoning.")


class ActionModel(BaseModel):
    """Typed action accepted by the environment."""

    action: str = Field(description="Action name, e.g., inspect, patch, run_tests, submit.")
    filename: Optional[str] = Field(default=None, description="Target filename for inspect/patch actions.")
    code: Optional[str] = Field(default=None, description="Replacement code for patch/submit actions.")


class RewardModel(BaseModel):
    """Structured reward payload with decomposition for RL diagnostics."""

    reward: float = Field(description="Final shaped reward in [0.0, 1.0].")
    tests_passed_ratio: float = Field(description="Fraction of tests passing after this step.")
    improvement_over_last_step: float = Field(description="Delta in pass ratio versus previous step.")
    step_penalty: float = Field(description="Penalty applied for current step count.")
    destructive_action_penalty: float = Field(description="Penalty for destructive edits.")
    components: Dict[str, float] = Field(default_factory=dict, description="Named reward terms for transparency.")


class ResetRequest(BaseModel):
    task: Optional[str] = Field(default=None, description="Task key: fix_broken_api, resolve_ci_pipeline, debug_hidden_state. If omitted, a random task is chosen.")
    difficulty: Optional[str] = Field(default=None, description="Difficulty level: easy, medium, hard. If omitted, defaults to medium.")


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
    score: float = 0.0
    internal: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class _EpisodeState:
    task: Any
    done: bool = False


class OpenEnvSWEEnv:
    """OpenEnv-compatible environment facade across all SWE-Sim tasks."""

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
        return ObservationModel(**task_obj.observation(last_reward=0.001))

    def step(self, action: ActionModel) -> Tuple[ObservationModel, RewardModel, bool, Dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._episode.done:
            raise RuntimeError("Episode already finished. Call reset() for a new episode.")

        reward_dict, done, info = self._episode.task.step(action.model_dump())
        self._episode.done = done

        observation = ObservationModel(**self._episode.task.observation(last_reward=reward_dict["reward"]))
        reward = RewardModel(**reward_dict)
        return observation, reward, done, info

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
            score=task.current_score,
            internal=task.state(),
        )
