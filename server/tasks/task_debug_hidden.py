from __future__ import annotations

from typing import Any, Dict, Tuple

from server.utils.code_runner import run_pytest_in_sandbox
from server.utils.graders import (
    _SCORE_MAX, _SCORE_MIN,
    guard_score, compute_destructive_penalty, safe_ratio,
)


class DebugHiddenStateTask:
    """HARD task — debug a 3-file codebase with hidden state and hidden tests."""

    name = "debug_hidden_state"

    def __init__(self) -> None:
        self.difficulty = "hard"
        self.steps_taken = 0
        self.max_steps = 14
        self.files: Dict[str, str] = {}
        self.error_log = "TypeError: NoneType object is not subscriptable (line 42)"
        # ── if/else guard at init: all ratios start at _SCORE_MIN, never raw 0.0 ──
        self.visible_ratio:       float = _SCORE_MIN
        self.hidden_ratio:        float = _SCORE_MIN
        self.last_combined_ratio: float = _SCORE_MIN
        self.current_score:       float = _SCORE_MIN

    def reset(self, difficulty: str = "hard") -> None:
        self.difficulty = difficulty
        self.steps_taken = 0
        self.max_steps = {"easy": 10, "medium": 12, "hard": 14}.get(difficulty, 14)
        self.files = self._broken_files(difficulty)

        visible = self._run_visible_tests()
        # safe_ratio() applied the if/else guard at the division site.
        self.visible_ratio = visible["ratio"]
        # ── if/else guard: hidden not yet evaluated → _SCORE_MIN, never raw 0.0 ──
        self.hidden_ratio        = _SCORE_MIN
        self.last_combined_ratio = guard_score(0.5 * self.visible_ratio + 0.5 * self.hidden_ratio)
        self.current_score       = _SCORE_MIN
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
            # ── if/else guard on last_reward ──
            "last_reward": guard_score(last_reward),
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
            # ── if/else guard on every ratio in state ──
            "visible_ratio": guard_score(self.visible_ratio),
            "hidden_ratio":  guard_score(self.hidden_ratio),
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        action_name = (action.get("action") or "").strip().lower()
        filename    = action.get("filename")
        code        = action.get("code")

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
            # ── if/else guard happens inside safe_ratio() ──
            self.visible_ratio = visible["ratio"]
            self.error_log     = visible["output"]
            info["visible_passed"] = str(visible["passed"])
            info["visible_total"]  = str(visible["total"])

        elif action_name == "submit":
            visible = self._run_visible_tests()
            hidden  = self._run_hidden_tests()
            # ── if/else guard happens inside safe_ratio() for both ──
            self.visible_ratio = visible["ratio"]
            self.hidden_ratio  = hidden["ratio"]
            self.error_log     = hidden["output"] or visible["output"]
            info["visible_passed"] = str(visible["passed"])
            info["visible_total"]  = str(visible["total"])
            info["hidden_passed"]  = str(hidden["passed"])
            info["hidden_total"]   = str(hidden["total"])
            done = True

        else:
            self.error_log = f"Unsupported action: {action_name}"

        # ── if/else guard on the combined ratio ──
        combined_ratio = guard_score(0.5 * self.visible_ratio + 0.5 * self.hidden_ratio)

        # Raw improvement is kept un-guarded for the formula so that negative
        # improvement correctly lowers the reward.  For dict fields we guard.
        improvement_raw     = combined_ratio - self.last_combined_ratio
        # ── if/else guard for the reward dict field (not the formula input) ──
        improvement_guarded = guard_score(improvement_raw)

        destructive_penalty = compute_destructive_penalty(self.files)

        base_reward = (
            0.5 * self.visible_ratio
            + 0.5 * self.hidden_ratio
            + 0.2 * improvement_raw      # raw — negative penalises correctly
            - 0.02 * self.steps_taken
            - destructive_penalty
        )
        # ── if/else guard on the final reward value ──
        reward_value = guard_score(base_reward)

        if self.steps_taken >= self.max_steps:
            done = True

        # ── if/else guard on every score assignment ──
        self.current_score       = combined_ratio   # already guarded above
        self.last_combined_ratio = combined_ratio

        reward = {
            # ── if/else guard on every reward dict field ──
            "reward":                     guard_score(reward_value),
            "tests_passed_ratio":         guard_score(combined_ratio),
            "improvement_over_last_step": improvement_guarded,
            "step_penalty":               guard_score(0.02 * self.steps_taken),
            "destructive_action_penalty": guard_score(destructive_penalty),
            "components": {
                # ── if/else guard on every component ──
                "visible":     guard_score(0.5 * self.visible_ratio),
                "hidden":      guard_score(0.5 * self.hidden_ratio),
                "improve":     guard_score(0.2 * abs(improvement_raw)),
                "step":        guard_score(0.02 * self.steps_taken),
                "destructive": guard_score(destructive_penalty),
            },
        }
        # ── if/else guard on info score ──
        info["score"] = guard_score(self.current_score)
        return reward, done, info

    def _visible_summary(self) -> str:
        total  = self._visible_total_for_difficulty()
        passed = int(round(self.visible_ratio * total))
        return f"{passed} passing, {total - passed} failing"

    def _visible_total_for_difficulty(self) -> int:
        return {"easy": 2, "medium": 3, "hard": 4}.get(self.difficulty, 4)

    def _run_visible_tests(self) -> Dict[str, Any]:
        tests  = {"test_visible.py": self._visible_tests(self.difficulty)}
        result = run_pytest_in_sandbox(files=self.files, tests=tests, timeout_seconds=10)
        passed = int(result["passed"])
        total  = int(result["total"])
        # ── safe_ratio() is the single division site — if/else guard inside ──
        return {"passed": passed, "total": total,
                "ratio": safe_ratio(passed, total), "output": result["output"]}

    def _run_hidden_tests(self) -> Dict[str, Any]:
        tests  = {"test_hidden.py": self._hidden_tests(self.difficulty)}
        result = run_pytest_in_sandbox(files=self.files, tests=tests, timeout_seconds=10)
        passed = int(result["passed"])
        total  = int(result["total"])
        # ── safe_ratio() is the single division site — if/else guard inside ──
        return {"passed": passed, "total": total,
                "ratio": safe_ratio(passed, total), "output": result["output"]}

    def _broken_files(self, difficulty: str) -> Dict[str, str]:
        config_code = (
            "DEFAULT_CONFIG = {'threshold': 10, 'window': 2}\n"
            "RUNTIME_CONFIG = dict(DEFAULT_CONFIG)\n\n"
            "def get_config():\n"
            "    return RUNTIME_CONFIG\n\n"
            "def reset_runtime():\n"
            "    global RUNTIME_CONFIG\n"
            "    RUNTIME_CONFIG = None  # BUG: should be dict(DEFAULT_CONFIG)\n"
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
            db_code = db_code.replace(
                "return [{'value': i} for i in range(limit)]",
                "return [{'value': i + 1} for i in range(limit)]",
            )
        return {"api.py": api_code, "db.py": db_code, "config.py": config_code}

    def _visible_tests(self, difficulty: str) -> str:
        if difficulty == "easy":
            return (
                "from api import compute_total\n\n"
                "def test_normal_call():\n    assert compute_total(6) == 9\n\n"
                "def test_negative_then_recovery():\n"
                "    compute_total(-1)\n    assert compute_total(6) == 9\n"
            )
        if difficulty == "medium":
            return (
                "from api import compute_total\n\n"
                "def test_normal_call():\n    assert compute_total(12) == 33\n\n"
                "def test_no_crash_negative():\n    assert compute_total(-1) == 0\n\n"
                "def test_recovery_after_negative():\n"
                "    compute_total(-1)\n    assert compute_total(12) == 33\n"
            )
        return (
            "from api import compute_total\n\n"
            "def test_normal_call():\n    assert compute_total(12) == 21\n\n"
            "def test_no_crash_negative():\n    assert compute_total(-1) == 0\n\n"
            "def test_recovery_after_negative():\n"
            "    compute_total(-1)\n    assert compute_total(12) == 21\n\n"
            "def test_multiple_calls_stable():\n"
            "    for _ in range(3):\n        assert compute_total(12) == 21\n"
        )

    def _hidden_tests(self, difficulty: str) -> str:
        if difficulty == "easy":
            return (
                "from api import compute_total\n\n"
                "def test_hidden_easy_consistency():\n"
                "    compute_total(-1)\n    assert compute_total(6) == 9\n"
            )
        if difficulty == "medium":
            return (
                "from api import compute_total\n\n"
                "def test_hidden_medium_sequence():\n"
                "    assert compute_total(15) >= 0\n"
                "    compute_total(-1)\n    assert compute_total(15) >= 0\n\n"
                "def test_hidden_medium_repeatability():\n"
                "    assert compute_total(10) >= 0\n"
            )
        return (
            "from api import compute_total\n\n"
            "def test_hidden_hard_sequence_1():\n"
            "    assert compute_total(20) == 145\n"
            "    compute_total(-1)\n    assert compute_total(20) == 145\n\n"
            "def test_hidden_hard_sequence_2():\n"
            "    for _ in range(5):\n        assert compute_total(12) == 21\n\n"
            "def test_hidden_hard_negative_guard():\n"
            "    assert compute_total(-5) == 0\n"
        )
