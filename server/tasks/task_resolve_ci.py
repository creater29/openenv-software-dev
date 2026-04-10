from __future__ import annotations

from typing import Any, Dict, Tuple

from server.utils.code_runner import run_pytest_in_sandbox
from server.utils.graders import clamp_score, compute_destructive_penalty, compute_shaped_reward


class ResolveCIPipelineTask:
    name = "resolve_ci_pipeline"

    def __init__(self) -> None:
        self.difficulty = "medium"
        self.steps_taken = 0
        self.max_steps = 10
        self.files: Dict[str, str] = {}
        self.error_log = ""
        self.last_pass_ratio = 0.0
        self.current_score = 0.0

    def reset(self, difficulty: str = "medium") -> None:
        self.difficulty = difficulty
        self.steps_taken = 0
        self.max_steps = {"easy": 8, "medium": 10, "hard": 12}.get(difficulty, 10)
        self.files = self._broken_files(difficulty)
        baseline = self._evaluate(self.files)
        self.last_pass_ratio = baseline["pass_ratio"]
        self.current_score = 0.0
        self.error_log = baseline["output"]

    def observation(self, last_reward: float) -> Dict[str, Any]:
        return {
            "task": self.name,
            "difficulty": self.difficulty,
            "step": self.steps_taken,
            "max_steps": self.max_steps,
            "files": self.files,
            "file": None,
            "code": None,
            "error_log": self.error_log,
            "test_results": self._summary(self.last_pass_ratio),
            "visible_tests": None,
            "hidden_tests": None,
            "last_reward": last_reward,
            "metadata": {
                "allowed_actions": ["inspect", "patch", "run_tests", "submit"],
                "weights": {
                    "W_pass": 1.0,
                    "W_improve": 0.3,
                    "W_step_penalty": 0.05,
                },
            },
        }

    def state(self) -> Dict[str, Any]:
        return {
            "files": self.files,
            "test_results": self._summary(self.last_pass_ratio),
            "error_log": self.error_log,
            "tests_passed_ratio": self.last_pass_ratio,
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        action_name = (action.get("action") or "").strip().lower()
        filename = action.get("filename")
        code = action.get("code")

        self.steps_taken += 1
        info: Dict[str, Any] = {"action": action_name}
        current_ratio = self.last_pass_ratio

        if action_name == "inspect":
            if filename not in self.files:
                self.error_log = f"Unknown file: {filename}"
            else:
                self.error_log = f"Inspected {filename}"
                info["content"] = self.files[filename]
        elif action_name == "patch":
            if filename not in self.files:
                self.error_log = f"Unknown file: {filename}"
            elif not isinstance(code, str):
                self.error_log = "Patch action requires a non-empty code payload"
            else:
                self.files[filename] = code
                self.error_log = f"Patched {filename}"
        elif action_name in {"run_tests", "submit"}:
            result = self._evaluate(self.files)
            current_ratio = result["pass_ratio"]
            self.error_log = result["output"]
            info["tests_passed"] = result["passed"]
            info["tests_total"] = result["total"]
        else:
            self.error_log = f"Unsupported action: {action_name}"

        improvement = current_ratio - self.last_pass_ratio
        destructive_penalty = compute_destructive_penalty(self.files)
        reward_value = compute_shaped_reward(
            tests_passed_ratio=current_ratio,
            improvement_over_last_step=improvement,
            steps_taken=self.steps_taken,
            destructive_action_penalty=destructive_penalty,
            w_pass=1.0,
            w_improve=0.3,
            w_step_penalty=0.05,
        )

        self.last_pass_ratio = current_ratio
        self.current_score = clamp_score(current_ratio)

        done = current_ratio >= 0.99 or self.steps_taken >= self.max_steps

        reward = {
            "reward": reward_value,
            "tests_passed_ratio": current_ratio,
            "improvement_over_last_step": improvement,
            "step_penalty": 0.05 * self.steps_taken,
            "destructive_action_penalty": destructive_penalty,
            "components": {
                "pass": 1.0 * current_ratio,
                "improve": 0.3 * improvement,
                "step": -(0.05 * self.steps_taken),
                "destructive": -destructive_penalty,
            },
        }
        info["score"] = self.current_score
        return reward, done, info

    def _evaluate(self, files: Dict[str, str]) -> Dict[str, Any]:
        result = run_pytest_in_sandbox(files=files, tests={"test_pipeline.py": self._tests(self.difficulty)}, timeout_seconds=10)
        passed = int(result["passed"])
        total = int(result["total"])
        pass_ratio = 0.0 if total == 0 else passed / total
        return {
            "passed": passed,
            "total": total,
            "pass_ratio": clamp_score(pass_ratio),
            "output": result["output"],
        }

    def _summary(self, ratio: float) -> str:
        total = 5 if self.difficulty == "hard" else 4
        passed = min(total, int(round(ratio * total)))
        return f"{passed} passing, {total - passed} failing"

    def _broken_files(self, difficulty: str) -> Dict[str, str]:
        if difficulty == "hard":
            utils_logic = (
                "def normalize(values):\n"
                "    if not values:\n"
                "        return []\n"
                "    total = len(values)\n"
                "    return [v / total for v in values]\n\n"
                "def moving_average(values):\n"
                "    if len(values) < 2:\n"
                "        return values\n"
                "    out = []\n"
                "    for i in range(len(values) - 1):\n"
                "        out.append((values[i] + values[i + 1]) / 2)\n"
                "    return out\n"
            )
            main_code = (
                "from utils import normalize, moving_average\n\n"
                "def pipeline(values):\n"
                "    smooth = moving_average(values)\n"
                "    return normalize(smooth)\n\n"
                "def quality_gate(values):\n"
                "    out = pipeline(values)\n"
                "    return round(sum(out), 3)\n"
            )
        else:
            utils_logic = (
                "def normalize(values):\n"
                "    if not values:\n"
                "        return []\n"
                "    total = len(values)\n"
                "    return [v / total for v in values]\n\n"
                "def moving_average(values):\n"
                "    if len(values) < 2:\n"
                "        return values\n"
                "    out = []\n"
                "    for i in range(len(values) - 1):\n"
                "        out.append((values[i] + values[i + 1]) / 2)\n"
                "    return out\n"
            )
            main_code = (
                "from utils import normalize, moving_average\n\n"
                "def pipeline(values):\n"
                "    smooth = moving_average(values)\n"
                "    return normalize(smooth)\n\n"
                "def quality_gate(values):\n"
                "    out = pipeline(values)\n"
                "    return round(sum(out), 6)\n"
            )
        return {"main.py": main_code, "utils.py": utils_logic}

    def _tests(self, difficulty: str) -> str:
        extra = "\n\ndef test_pipeline_balanced_case():\n    assert quality_gate([2, 2, 2, 2]) == 1.0\n" if difficulty == "hard" else ""
        return (
            "from main import pipeline, quality_gate\n\n"
            "def test_pipeline_sum_is_one():\n"
            "    result = pipeline([1, 2, 3, 4])\n"
            "    assert round(sum(result), 6) == 1.0\n\n"
            "def test_pipeline_max_is_bounded():\n"
            "    result = pipeline([1, 2, 3, 4])\n"
            "    assert max(result) <= 1.0\n\n"
            "def test_pipeline_values_ordered():\n"
            "    result = pipeline([1, 2, 3, 4])\n"
            "    assert result[0] < result[-1]\n\n"
            "def test_quality_gate_sum():\n"
            "    assert quality_gate([1, 2, 3, 4]) == 1.0\n\n"
            "def test_empty_pipeline():\n"
            "    assert pipeline([]) == []\n"
            f"{extra}"
        )
