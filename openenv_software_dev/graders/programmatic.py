"""
Programmatic grader — deterministic scoring via pytest execution.

This is the primary grader.  It runs the test suite inside the sandbox and
returns a normalised score based on the fraction of tests that pass.
"""
from typing import Any, Dict

from ..sandbox.executor import SandboxedExecutor
from ..sandbox.filesystem import VirtualFilesystem


class ProgrammaticGrader:
    """
    Scores the agent's solution by running pytest and parsing the results.

    Scoring:
      tests_passed  = passed / total          (0.0 – 1.0)
      code_quality  = 1.0 if no syntax errors, else 0.5 or 0.0
      score         = weighted combination
    """

    def __init__(self, test_weight: float = 0.8, quality_weight: float = 0.2):
        self.test_weight = test_weight
        self.quality_weight = quality_weight

    def grade(
        self,
        task,
        vfs: VirtualFilesystem,
        executor: SandboxedExecutor,
    ) -> Dict[str, Any]:
        # ── Run tests ──────────────────────────────────────────────────
        test_result = executor.run_tests(vfs)
        tests_passed = test_result.get("tests_passed", 0.0)
        passed = test_result.get("passed", 0)
        failed = test_result.get("failed", 0)
        total  = test_result.get("total", 0)

        # ── Static acceptance check ────────────────────────────────────
        acceptance = task.acceptance_check(vfs.snapshot())
        accepted   = acceptance.get("accepted", False)

        # ── Code quality proxy (syntax check via acceptance) ───────────
        code_quality = 1.0 if accepted else (0.5 if tests_passed > 0 else 0.0)

        # ── Composite score ────────────────────────────────────────────
        score = (
            self.test_weight    * tests_passed
            + self.quality_weight * code_quality
        )

        return {
            "score":        round(score, 4),
            "tests_passed": tests_passed,
            "code_quality": code_quality,
            "accepted":     accepted,
            "passed":       passed,
            "failed":       failed,
            "total":        total,
            "test_output":  test_result.get("output", ""),
            "checks":       acceptance.get("checks", []),
            "grader":       "programmatic",
        }
