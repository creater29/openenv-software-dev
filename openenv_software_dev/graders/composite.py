"""
Composite grader — merges programmatic and optional LLM scores.

The final score is a weighted average:
  score = programmatic_weight * prog_score + llm_weight * llm_score

When LLM grading is disabled, the programmatic score is used directly.
"""
from typing import Any, Dict, Optional

from .programmatic import ProgrammaticGrader
from .llm_grader    import LLMGrader


class CompositeGrader:
    """
    Orchestrates one or two graders and returns a unified grading result.

    Parameters
    ----------
    enable_llm          : bool   — whether to call the LLM grader.
    llm_endpoint        : str    — chat-completions URL (OpenAI-compatible).
    programmatic_weight : float  — weight given to the deterministic grader.
    llm_weight          : float  — weight given to the LLM grader.
    """

    def __init__(
        self,
        enable_llm: bool = False,
        llm_endpoint: Optional[str] = None,
        programmatic_weight: float = 0.7,
        llm_weight: float = 0.3,
    ):
        self.prog_grader = ProgrammaticGrader()
        self.llm_grader  = LLMGrader(endpoint=llm_endpoint) if (enable_llm and llm_endpoint) else None
        self.prog_w      = programmatic_weight
        self.llm_w       = llm_weight

    def grade(self, task, vfs, executor) -> Dict[str, Any]:
        """Run all enabled graders and return a merged result dict."""

        # ── Programmatic (always runs) ─────────────────────────────────
        prog = self.prog_grader.grade(task, vfs, executor)
        final_score = prog["score"]

        llm_result = {}

        # ── LLM (optional) ─────────────────────────────────────────────
        if self.llm_grader is not None:
            llm_result  = self.llm_grader.grade(task, vfs, executor)
            final_score = (
                self.prog_w * prog["score"]
                + self.llm_w * llm_result.get("score", 0.5)
            )

        return {
            # Core fields consumed by RewardCalculator
            "score":        round(final_score, 4),
            "tests_passed": prog["tests_passed"],
            "code_quality": prog["code_quality"],
            "accepted":     prog["accepted"],
            # Diagnostics
            "passed":       prog["passed"],
            "failed":       prog["failed"],
            "total":        prog["total"],
            "test_output":  prog["test_output"],
            "checks":       prog["checks"],
            "programmatic": prog,
            "llm":          llm_result,
        }
