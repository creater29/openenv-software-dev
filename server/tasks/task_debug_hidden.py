from __future__ import annotations

from typing import Any, Dict, Tuple

from server.utils.code_runner import run_pytest_in_sandbox
from server.utils.graders import clamp_score, compute_destructive_penalty, sanitize_reward_dict


class DebugHiddenStateTask:
    name = "debug_hidden_state"

    def __init__(self) -> None:
        self.difficulty = "hard"
        self.steps_taken = 0
        self.max_steps = 14
        self.files: Dict[str, str] = {}
        self.error_log = "TypeError: NoneType object is not subscriptable (line 42)"
        self.visible_ratio = 0.0
        self.hidden_ratio = 0.0
        self.last_combined_ratio = 0.0
        self.current_score = 0.0

    def reset(self, difficulty: str = "hard") -> None:
        self.difficulty = difficulty
        self.steps_taken = 0
        self.max_steps = {"easy": 10, "medium": 12, "hard": 14}.get(difficulty, 14)
        self.files = self._broken_files(difficulty)

        visible = self._run_visible_tests()
        self.visible_ratio = visible["ratio"]
        self.hidden_ratio = 0.0
        self.last_combined_ratio = 0.5 * self.visible_ratio
        self.current_score = 0.0
        self.error_log = visible["output"] or "TypeError: NoneType object is not subscriptable (line 42)"

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
            "test_results": None,
            "visible_tests": self._visible_summary(),
            "hidden_tests": "[HIDDEN - evaluated at submit]",
            "last_reward": last_reward,
            "metadata": {
                "allowed_actions": ["inspect", "patch", "run_tests", "submit"],
                "weights": {
                    "W_visible": 0.5,
                    "W_hidden": 0.5,
                    "W_improve": 0.2,
                    "W_step_penalty": 0.02,
                },
            },
        }

    def state(self) -> Dict[str, Any]:
        return {
            "files": self.files,
            "error_log": self.error_log,
            "visible_tests": self._visible_summary(),
            "hidden_tests": "[HIDDEN - evaluated at submit]",
            "visible_ratio": self.visible_ratio,
            "hidden_ratio": self.hidden_ratio,
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        action_name = (action.get("action") or "").strip().lower()
        filename = action.get("filename")
        code = action.get("code")

        self.steps_taken += 1
        info: Dict[str, Any] = {"action": action_name}
        done = False

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
                self.error_log = "Patch requires replacement code"
            else:
                self.files[filename] = code
                self.error_log = f"Patched {filename}"
        elif action_name == "run_tests":
            visible = self._run_visible_tests()
            self.visible_ratio = visible["ratio"]
            self.error_log = visible["output"]
            info["visible_passed"] = visible["passed"]
            info["visible_total"] = visible["total"]
        elif action_name == "submit":
            visible = self._run_visible_tests()
            hidden = self._run_hidden_tests()
            self.visible_ratio = visible["ratio"]
            self.hidden_ratio = hidden["ratio"]
            self.error_log = hidden["output"] or visible["output"]
            info["visible_passed"] = visible["passed"]
            info["visible_total"] = visible["total"]
            info["hidden_passed"] = hidden["passed"]
            info["hidden_total"] = hidden["total"]
            done = True
        else:
            self.error_log = f"Unsupported action: {action_name}"

        combined_ratio = clamp_score((self.visible_ratio + self.hidden_ratio) / 2)
        improvement = combined_ratio - self.last_combined_ratio
        destructive_penalty = compute_destructive_penalty(self.files)
        base_reward = (0.5 * self.visible_ratio) + (0.5 * self.hidden_ratio) + (0.2 * improvement) - (0.02 * self.steps_taken) - destructive_penalty
        reward_value = clamp_score(base_reward)

        if self.steps_taken >= self.max_steps:
            done = True

        self.current_score = combined_ratio
        self.last_combined_ratio = combined_ratio

        reward = {
            "reward": reward_value,
            "tests_passed_ratio": combined_ratio,
            "improvement_over_last_step": improvement,
            "step_penalty": 0.02 * self.steps_taken,
            "destructive_action_penalty": destructive_penalty,
            "components": {
                "visible": 0.5 * self.visible_ratio,
                "hidden": 0.5 * self.hidden_ratio,
                "improve": 0.2 * improvement,
                "step": -(0.02 * self.steps_taken),
                "destructive": -destructive_penalty,
            },
        }
        info["score"] = self.current_score
        return sanitize_reward_dict(reward), done, info

    def _visible_summary(self) -> str:
        total = self._visible_total_for_difficulty()
        passed = int(round(self.visible_ratio * total))
        return f"{passed} passing, {total - passed} failing"

    def _visible_total_for_difficulty(self) -> int:
        return {"easy": 2, "medium": 3, "hard": 4}.get(self.difficulty, 4)

    def _run_visible_tests(self) -> Dict[str, Any]:
        tests = {"test_visible.py": self._visible_tests(self.difficulty)}
        result = run_pytest_in_sandbox(files=self.files, tests=tests, timeout_seconds=10)
        passed = int(result["passed"])
        total = int(result["total"])
        ratio = 0.0 if total == 0 else passed / total
        return {"passed": passed, "total": total, "ratio": clamp_score(ratio), "output": result["output"]}

    def _run_hidden_tests(self) -> Dict[str, Any]:
        tests = {"test_hidden.py": self._hidden_tests(self.difficulty)}
        result = run_pytest_in_sandbox(files=self.files, tests=tests, timeout_seconds=10)
        passed = int(result["passed"])
        total = int(result["total"])
        ratio = 0.0 if total == 0 else passed / total
        return {"passed": passed, "total": total, "ratio": clamp_score(ratio), "output": result["output"]}

    def _broken_files(self, difficulty: str) -> Dict[str, str]:
        config_code = (
            "DEFAULT_CONFIG = {'threshold': 10, 'window': 2}\n"
            "RUNTIME_CONFIG = dict(DEFAULT_CONFIG)\n\n"
            "def get_config():\n"
            "    return RUNTIME_CONFIG\n\n"
            "def reset_runtime():\n"
            "    global RUNTIME_CONFIG\n"
            "    RUNTIME_CONFIG = None\n"
        )

        db_code = (
            "from config import get_config, reset_runtime\n\n"
            "def fetch_records(limit):\n"
            "    if limit < 0:\n"
            "        reset_runtime()\n"
            "        return []\n"
            "    return [{'value': i} for i in range(limit)]\n\n"
            "def thresholded_values(limit):\n"
            "    cfg = get_config()\n"
            "    rows = fetch_records(limit)\n"
            "    return [r['value'] for r in rows if r['value'] >= cfg['threshold']]\n"
        )

        api_code = (
            "from db import thresholded_values\n\n"
            "def compute_total(limit):\n"
            "    values = thresholded_values(limit)\n"
            "    return sum(values)\n"
        )

        if difficulty == "easy":
            config_code = config_code.replace("'threshold': 10", "'threshold': 4")
        if difficulty == "medium":
            db_code = db_code.replace("return [{'value': i} for i in range(limit)]", "return [{'value': i + 1} for i in range(limit)]")

        return {"api.py": api_code, "db.py": db_code, "config.py": config_code}

    def _visible_tests(self, difficulty: str) -> str:
        if difficulty == "easy":
            return (
                "from api import compute_total\n\n"
                "def test_normal_call():\n"
                "    assert compute_total(6) == 9\n\n"
                "def test_negative_then_recovery():\n"
                "    compute_total(-1)\n"
                "    assert compute_total(6) == 9\n"
            )
        if difficulty == "medium":
            return (
                "from api import compute_total\n\n"
                "def test_normal_call():\n"
                "    assert compute_total(12) == 33\n\n"
                "def test_no_crash_negative():\n"
                "    assert compute_total(-1) == 0\n\n"
                "def test_recovery_after_negative():\n"
                "    compute_total(-1)\n"
                "    assert compute_total(12) == 33\n"
            )
        return (
            "from api import compute_total\n\n"
            "def test_normal_call():\n"
            "    assert compute_total(12) == 21\n\n"
            "def test_no_crash_negative():\n"
            "    assert compute_total(-1) == 0\n\n"
            "def test_recovery_after_negative():\n"
            "    compute_total(-1)\n"
            "    assert compute_total(12) == 21\n\n"
            "def test_multiple_calls_stable():\n"
            "    for _ in range(3):\n"
            "        assert compute_total(12) == 21\n"
        )

    def _hidden_tests(self, difficulty: str) -> str:
        if difficulty == "easy":
            return (
                "from api import compute_total\n\n"
                "def test_hidden_easy_consistency():\n"
                "    compute_total(-1)\n"
                "    assert compute_total(6) == 9\n"
            )
        if difficulty == "medium":
            return (
                "from api import compute_total\n\n"
                "def test_hidden_medium_sequence():\n"
                "    assert compute_total(15) >= 0\n"
                "    compute_total(-1)\n"
                "    assert compute_total(15) >= 0\n\n"
                "def test_hidden_medium_repeatability():\n"
                "    assert compute_total(10) >= 0\n"
            )
        return (
            "from api import compute_total\n\n"
            "def test_hidden_hard_sequence_1():\n"
            "    assert compute_total(20) == 145\n"
            "    compute_total(-1)\n"
            "    assert compute_total(20) == 145\n\n"
            "def test_hidden_hard_sequence_2():\n"
            "    for _ in range(5):\n"
            "        assert compute_total(12) == 21\n\n"
            "def test_hidden_hard_negative_guard():\n"
            "    assert compute_total(-5) == 0\n"
        )
