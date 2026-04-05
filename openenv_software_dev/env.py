"""
Core SoftwareDevEnv — a Gymnasium-compatible RL environment for software development.

The agent loop:
  obs, info = env.reset()
  while not done:
      action = agent.act(obs)          # dict with 'type', 'target_file', 'text_input'
      obs, reward, done, trunc, info = env.step(action)

Reward formula (per step):
  R = (tests_passed * 1.0) + (code_quality * 0.5) - 0.02
On SUBMIT a terminal bonus is added:
  R += 5.0 * grading_score
"""
import gymnasium as gym
from typing import Any, Optional, Dict

from .actions import ActionRegistry, ActionType
from .observation import ObservationBuilder
from .reward import RewardCalculator
from .sandbox.filesystem import VirtualFilesystem
from .sandbox.executor import SandboxedExecutor
from .graders.composite import CompositeGrader
from .task_registry import TaskRegistry


class SoftwareDevEnv(gym.Env):
    """
    OpenEnv-compliant Gymnasium environment for software engineering tasks.

    Agents can read/write files in a virtual filesystem, run tests, and submit
    a final solution.  Tasks include bug fixes, feature implementations, and
    code-quality challenges.  Grading uses deterministic (pytest-based)
    evaluation plus optional LLM-based semantic scoring.
    """

    metadata = {
        "render_modes": ["human", "ansi", "json"],
        "openenv_version": "0.1.0",
        "task_domain": "software_development",
    }

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        task_ids: Optional[list] = None,
        max_steps: int = 30,
        difficulty: str = "medium",
        enable_llm_grading: bool = False,
        llm_endpoint: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.enable_llm_grading = enable_llm_grading
        self.llm_endpoint = llm_endpoint

        # Sub-systems
        self.registry = TaskRegistry()
        self.registry.load_defaults(difficulty=difficulty)

        self.fs = VirtualFilesystem()
        self.executor = SandboxedExecutor()
        self.obs_builder = ObservationBuilder()
        self.reward_calc = RewardCalculator()
        self.grader = CompositeGrader(
            enable_llm=enable_llm_grading, llm_endpoint=llm_endpoint
        )

        # Episode state
        self.current_task = None
        self.steps = 0
        self.history = []

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode: sample a task and reset the virtual filesystem."""
        super().reset(seed=seed)
        self.steps = 0
        self.history = []

        # Sample a task (reproducible if seed provided)
        self.current_task = self.registry.sample(seed=seed)

        # Initialise the virtual FS with starter files for this task
        self.fs = VirtualFilesystem()
        for path, content in self.current_task.starter_files.items():
            self.fs.write(path, content)

        obs = self.obs_builder.build(
            self.current_task, self.fs, "initialized", 0.0, 0.0
        )
        info = {
            "task_id": self.current_task.task_id,
            "task_description": self.current_task.description,
            "difficulty": self.current_task.difficulty,
        }
        return obs, info

    def step(self, action_dict: Dict[str, Any]):
        """
        Execute one agent action and return (obs, reward, terminated, truncated, info).

        action_dict keys:
          - type        : int or ActionType  (required)
          - target_file : str               (required for READ_FILE / WRITE_FILE)
          - text_input  : str               (required for WRITE_FILE)
        """
        self.steps += 1

        action_type  = action_dict.get("type")
        target_file  = action_dict.get("target_file", "")
        text_input   = action_dict.get("text_input", "")

        # 1. Execute the chosen action
        action_result = self._execute_action(action_type, target_file, text_input)
        self.history.append({"step": self.steps, "action": action_dict, "result": action_result})

        # 2. Grade current solution state
        grading_result = self.grader.grade(self.current_task, self.fs, self.executor)

        # 3. Check termination
        submitted   = (int(action_type) == int(ActionType.SUBMIT))
        truncated   = (self.steps >= self.max_steps) and not submitted
        terminated  = submitted

        # 4. Compute reward
        reward = self.reward_calc.compute(
            action_result=action_result,
            grading_result=grading_result,
            step=self.steps,
            max_steps=self.max_steps,
            is_terminal=terminated,
        )

        # 5. Build observation
        progress = self.steps / self.max_steps
        obs = self.obs_builder.build(
            self.current_task, self.fs, action_result["status"], progress, reward
        )

        info = {
            "grading": grading_result,
            "action_result": action_result,
            "step": self.steps,
        }
        return obs, reward, terminated, truncated, info

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "ansi" or self.render_mode == "human":
            if self.current_task is None:
                return "No active episode. Call reset() first."
            snapshot = self.fs.snapshot()
            lines = [
                f"Task: {self.current_task.task_id}",
                f"Step: {self.steps}/{self.max_steps}",
                "Files:",
            ]
            for fname, content in snapshot.items():
                lines.append(f"  [{fname}]")
                for line in content.splitlines():
                    lines.append(f"    {line}")
            return "\n".join(lines)
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _execute_action(
        self, action_type, target_file: str, text_input: str
    ) -> Dict[str, Any]:
        """Dispatch the action to the appropriate handler and return a result dict."""
        try:
            atype = int(action_type)
        except (TypeError, ValueError):
            return {"status": "error", "msg": f"Invalid action type: {action_type}"}

        if atype == ActionType.READ_FILE:
            content = self.fs.read(target_file)
            return {"status": "success", "content": content}

        elif atype == ActionType.WRITE_FILE:
            if not target_file:
                return {"status": "error", "msg": "target_file is required for WRITE_FILE"}
            self.fs.write(target_file, text_input)
            return {"status": "success", "msg": f"Wrote {len(text_input)} chars to {target_file}"}

        elif atype == ActionType.RUN_TESTS:
            return self.executor.run_tests(self.fs)

        elif atype == ActionType.SUBMIT:
            return {"status": "submitted", "msg": "Solution submitted for final grading."}

        elif atype == ActionType.LIST_FILES:
            return {"status": "success", "files": sorted(self.fs.snapshot().keys())}

        return {"status": "error", "msg": f"Unknown action type: {action_type}"}
