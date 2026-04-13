from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from server.tasks.task_debug_hidden import DebugHiddenStateTask
from server.tasks.task_fix_api import FixBrokenApiTask
from server.tasks.task_resolve_ci import ResolveCIPipelineTask
from server.utils.graders import guard_score, sanitize_any


class ObservationModel(BaseModel):
    """Typed environment observation returned at every transition."""
    task: str = Field(description="Current task identifier.")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard.")
    step: int = Field(description="Current step count in this episode.")
    max_steps: int = Field(description="Maximum allowed steps for this episode.")
    files: Optional[Dict[str, str]] = Field(default=None)
    file: Optional[str] = Field(default=None)
    code: Optional[str] = Field(default=None)
    error_log: Optional[str] = Field(default=None)
    test_results: Optional[str] = Field(default=None)
    visible_tests: Optional[str] = Field(default=None)
    hidden_tests: Optional[str] = Field(default=None)
    last_reward: float = Field(description="Reward from previous step — always in (0,1).")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionModel(BaseModel):
    """Typed action accepted by the environment."""
    action: str = Field(description="Action name: inspect, patch, run_tests, submit.")
    filename: Optional[str] = Field(default=None)
    code: Optional[str] = Field(default=None)


class RewardModel(BaseModel):
    """Structured reward with full decomposition — every float in (0, 1)."""
    reward: float = Field(description="Shaped reward strictly in (0, 1).")
    tests_passed_ratio: float = Field(description="Test pass fraction, clamped to (0, 1).")
    improvement_over_last_step: float = Field(description="Delta in pass ratio vs previous step.")
    step_penalty: float = Field(description="Penalty for current step count.")
    destructive_action_penalty: float = Field(description="Penalty for destructive edits.")
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
    # ── if/else guard at default: never 0.0 ──
    score: float = 0.001
    internal: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class _EpisodeState:
    task: Any
    done: bool = False


# ── Builder helpers — final sanitization layer before Pydantic ───────────────

def _build_observation(task_obj: Any, last_reward: float) -> ObservationModel:
    """Build ObservationModel with guard_score() applied via sanitize_any()
    on every float in the entire observation dict recursively.
    """
    raw   = task_obj.observation(last_reward=last_reward)
    # sanitize_any() calls guard_score() (the if/else gate) on every float
    # found anywhere in the dict tree — weights, last_reward, ratios, all of it.
    clean = sanitize_any(raw)
    return ObservationModel(**clean)


def _build_reward(reward_dict: Dict[str, Any]) -> RewardModel:
    """Build RewardModel with guard_score() applied via sanitize_any()
    on every float, including all nested component values.
    """
    clean = sanitize_any(reward_dict)
    return RewardModel(**clean)


class OpenEnvSWEEnv:
    """OpenEnv-compatible environment facade across all SWE-Sim tasks."""

    def __init__(self) -> None:
        self._episode: Optional[_EpisodeState] = None
        self._task_registry = {
            "fix_broken_api":      FixBrokenApiTask,
            "resolve_ci_pipeline": ResolveCIPipelineTask,
            "debug_hidden_state":  DebugHiddenStateTask,
        }

    def reset(self, task: str, difficulty: str = "medium") -> ObservationModel:
        if task not in self._task_registry:
            raise ValueError(f"Unknown task: {task}")
        task_obj = self._task_registry[task]()
        task_obj.reset(difficulty=difficulty)
        self._episode = _EpisodeState(task=task_obj, done=False)
        # ── if/else guard: pass 0.001 not 0.0 as initial last_reward ──
        return _build_observation(task_obj, last_reward=0.001)

    def step(self, action: ActionModel) -> Tuple[ObservationModel, RewardModel, bool, Dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._episode.done:
            raise RuntimeError("Episode already finished. Call reset() for a new episode.")

        reward_dict, done, info = self._episode.task.step(action.model_dump())
        self._episode.done = done

        observation = _build_observation(self._episode.task, last_reward=reward_dict["reward"])
        reward      = _build_reward(reward_dict)
        # sanitize_any() guards every float in info too (e.g. the "score" field).
        clean_info  = sanitize_any(info)
        return observation, reward, done, clean_info

    def state(self) -> StateResponse:
        if self._episode is None:
            return StateResponse(ready=False)

        task      = self._episode.task
        raw_score = task.current_score

        # ── explicit if/else guard: 0 → 0.001, 1 → 0.999, else unchanged ──
        if raw_score is None:
            safe_score = 0.001
        elif raw_score <= 0.0:    # if score is 0 → give 0.001
            safe_score = 0.001
        elif raw_score >= 1.0:    # elif score is 1 → give 0.999
            safe_score = 0.999
        else:                     # else → use as-is (interior value)
            safe_score = float(raw_score)

        return StateResponse(
            ready=True,
            task=task.name,
            difficulty=task.difficulty,
            step=task.steps_taken,
            max_steps=task.max_steps,
            done=self._episode.done,
            score=safe_score,
            # sanitize_any() guards every float inside the internal state dict.
            internal=sanitize_any(task.state()),
        )
