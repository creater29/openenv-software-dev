from __future__ import annotations

from typing import Any, Dict, List, Tuple

from server.utils.code_runner import run_pytest_in_sandbox
from server.utils.graders import _SCORE_MAX, _SCORE_MIN, clamp_score, compute_destructive_penalty, compute_shaped_reward


class FixBrokenApiTask:
    name = "fix_broken_api"

    def __init__(self) -> None:
        self.difficulty = "medium"
        self.steps_taken = 0
        self.max_steps = 5
        self.file_name = "app.py"
        self.current_code = ""
        self.error_log = ""
        self.last_pass_ratio = _SCORE_MIN   # never raw 0.0
        self.current_score = _SCORE_MIN     # never raw 0.0

    def reset(self, difficulty: str = "medium") -> None:
        self.difficulty = difficulty
        self.steps_taken = 0
        self.max_steps = {"easy": 4, "medium": 5, "hard": 6}.get(difficulty, 5)
        self.current_code = self._broken_code(difficulty)
        baseline = self._evaluate_code(self.current_code)
        self.last_pass_ratio = baseline["pass_ratio"]   # already clamped by _evaluate_code
        self.current_score = _SCORE_MIN                 # never raw 0.0
        self.error_log = baseline["output"] or "NameError in route handler"

    def observation(self, last_reward: float) -> Dict[str, Any]:
        return {
            "task": self.name,
            "difficulty": self.difficulty,
            "step": self.steps_taken,
            "max_steps": self.max_steps,
            "file": self.file_name,
            "code": self.current_code,
            "error_log": self.error_log,
            "test_results": self._test_summary(self.last_pass_ratio),
            "files": None,
            "visible_tests": None,
            "hidden_tests": None,
            "last_reward": last_reward,
            "metadata": {
                "allowed_actions": ["inspect", "submit"],
                "weights": {
                    "W_pass": 0.75,
                    "W_improve": 0.25,
                    "W_step_penalty": 0.03,
                },
            },
        }

    def state(self) -> Dict[str, Any]:
        return {
            "file": self.file_name,
            "code": self.current_code,
            "error": self.error_log,
            "tests_passed_ratio": self.last_pass_ratio,
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        action_name = (action.get("action") or "").strip().lower()
        submitted_code = action.get("code")

        self.steps_taken += 1
        info: Dict[str, Any] = {"action": action_name}

        current_ratio = self.last_pass_ratio
        if action_name == "inspect":
            self.error_log = self.error_log or "Inspecting current stack trace"
        elif action_name == "submit":
            if not isinstance(submitted_code, str) or not submitted_code.strip():
                self.error_log = "Empty submission received"
            else:
                self.current_code = submitted_code
                result = self._evaluate_code(self.current_code)
                current_ratio = result["pass_ratio"]
                self.error_log = result["output"]
                info["tests_passed"] = result["passed"]
                info["tests_total"] = result["total"]
        else:
            self.error_log = f"Unsupported action for {self.name}: {action_name}"

        improvement = current_ratio - self.last_pass_ratio
        destructive_penalty = compute_destructive_penalty({self.file_name: self.current_code})
        reward_value = compute_shaped_reward(
            tests_passed_ratio=current_ratio,
            improvement_over_last_step=improvement,
            steps_taken=self.steps_taken,
            destructive_action_penalty=destructive_penalty,
            w_pass=0.75,
            w_improve=0.25,
            w_step_penalty=0.03,
        )
        self.last_pass_ratio = current_ratio
        self.current_score = clamp_score(current_ratio)

        done = current_ratio >= _SCORE_MAX or self.steps_taken >= self.max_steps

        reward = {
            "reward": reward_value,
            "tests_passed_ratio": clamp_score(current_ratio),
            "improvement_over_last_step": clamp_score(improvement),
            "step_penalty": clamp_score(0.03 * self.steps_taken),
            "destructive_action_penalty": clamp_score(destructive_penalty),
            "components": {
                "pass": clamp_score(0.75 * current_ratio),
                "improve": clamp_score(0.25 * abs(improvement)),
                "step": clamp_score(0.03 * self.steps_taken),
                "destructive": clamp_score(destructive_penalty),
            },
        }
        info["score"] = self.current_score
        return reward, done, info

    def _evaluate_code(self, code: str) -> Dict[str, Any]:
        files = {self.file_name: code}
        tests = {"test_app.py": self._hidden_tests(self.difficulty)}
        result = run_pytest_in_sandbox(files=files, tests=tests, timeout_seconds=8)
        passed = int(result["passed"])
        total = int(result["total"])
        pass_ratio = 0.0 if total == 0 else passed / total
        return {
            "passed": passed,
            "total": total,
            "pass_ratio": clamp_score(pass_ratio),
            "output": result["output"],
        }

    def _test_summary(self, ratio: float) -> str:
        passed = int(round(ratio * 3))
        return f"{passed} passing, {3 - passed} failing"

    def _broken_code(self, difficulty: str) -> str:
        if difficulty == "easy":
            return (
                "from fastapi import FastAPI\n\n"
                "app = FastAPI()\n\n"
                "def compute_total(items):\n"
                "    return sum(item)\n\n"
                "@app.get('/total')\n"
                "def total_route():\n"
                "    values = [1, 2, 3]\n"
                "    return {'total': compute_total(values)}\n"
            )
        if difficulty == "hard":
            return (
                "from fastapi import FastAPI\n\n"
                "app = FastAPI()\n\n"
                "def compute_total(items):\n"
                "    if not items:\n"
                "        return 1\n"
                "    total = sum(items)\n"
                "    return total - 1\n\n"
                "@app.get('/total')\n"
                "def total_route():\n"
                "    values = [1, 2, 3]\n"
                "    return {'total': compute_total(values)}\n"
            )
        return (
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n"
            "def compute_total(items):\n"
            "    return sum(items) - 1\n\n"
            "@app.get('/total')\n"
            "def total_route():\n"
            "    values = [1, 2, 3]\n"
            "    return {'total': compute_total(values)}\n"
        )

    def _hidden_tests(self, difficulty: str) -> str:
        # Exactly 3 deterministic hidden tests per spec.
        if difficulty == "hard":
            checks: List[str] = [
                "assert compute_total([1, 2, 3]) == 6",
                "assert compute_total([]) == 0",
                "assert compute_total([5, -2, 4]) == 7",
            ]
        elif difficulty == "easy":
            checks = [
                "assert compute_total([1, 2, 3]) == 6",
                "assert compute_total([10]) == 10",
                "assert compute_total([]) == 0",
            ]
        else:
            checks = [
                "assert compute_total([4, 6]) == 10",
                "assert compute_total([0, 0, 0]) == 0",
                "assert compute_total([-1, 1, 2]) == 2",
            ]
        return (
            "from app import compute_total\n\n"
            "def test_case_1():\n"
            f"    {checks[0]}\n\n"
            "def test_case_2():\n"
            f"    {checks[1]}\n\n"
            "def test_case_3():\n"
            f"    {checks[2]}\n"
        )
