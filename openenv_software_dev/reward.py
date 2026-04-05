"""
Reward shaping logic for the SoftwareDev environment.

Reward formula:
  R = (tests_passed * 1.0) + (code_quality * 0.5) - (step_penalty * 0.02)

On a SUBMIT action a terminal bonus is added:
  R += terminal_weight * grading_score
"""
from typing import Any, Dict


class RewardCalculator:
    """
    Computes per-step and terminal rewards from the grading result.

    Weights can be tuned via the config; these defaults follow the
    specification in the README benchmark section.
    """

    def __init__(
        self,
        test_weight: float = 1.0,
        quality_weight: float = 0.5,
        step_penalty: float = 0.02,
        terminal_weight: float = 5.0,
    ):
        self.test_weight = test_weight
        self.quality_weight = quality_weight
        self.step_penalty = step_penalty
        self.terminal_weight = terminal_weight

    def compute(
        self,
        action_result: Dict[str, Any],
        grading_result: Dict[str, Any],
        step: int,
        max_steps: int,
        is_terminal: bool,
    ) -> float:
        """Return the shaped reward for a single environment step."""

        # ── Base components ────────────────────────────────────────────
        tests_passed = grading_result.get("tests_passed", 0.0)   # 0–1 fraction
        code_quality = grading_result.get("code_quality", 0.0)   # 0–1 score
        action_ok = 1.0 if action_result.get("status") == "success" else 0.0

        reward = (
            self.test_weight * tests_passed
            + self.quality_weight * code_quality
            - self.step_penalty            # flat per-step cost to incentivise efficiency
        )

        # ── Terminal bonus (only on explicit SUBMIT) ───────────────────
        if is_terminal:
            final_score = grading_result.get("score", 0.0)
            reward += self.terminal_weight * final_score

        return round(reward, 4)
