"""
Bug-fix tasks — the agent must identify and correct a deliberate defect.

The `acceptance_check` method performs a lightweight static check (substring
search) in addition to the programmatic grader that runs pytest.  This lets the
environment award partial credit even before tests are run, and also serves as
a sanity gate in case pytest is unavailable.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List
from .base import Task


@dataclass
class BugFixTask(Task):
    """
    A task where the starter code contains an intentional bug.
    The agent must locate the bug and fix it so the test suite passes.
    """

    category: str = "bug_fix"

    # Default starter files for the simplest bug (subtraction instead of addition)
    starter_files: Dict[str, str] = field(
        default_factory=lambda: {
            "solution.py": (
                "def add(a, b):\n"
                "    # BUG: should be a + b\n"
                "    return a - b\n"
            ),
            "test_solution.py": (
                "from solution import add\n\n"
                "def test_add_positive():\n"
                "    assert add(1, 2) == 3\n\n"
                "def test_add_zero():\n"
                "    assert add(0, 5) == 5\n"
            ),
        }
    )

    # Patterns that indicate the bug has been fixed (for static check)
    correct_patterns: List[str] = field(
        default_factory=lambda: ["return a + b"]
    )

    def acceptance_check(self, snapshot: Dict[str, str]) -> Dict[str, Any]:
        """
        Two-stage acceptance check:
          1. Static pattern match — fast, no subprocess.
          2. Returns a structured result that CompositeGrader can merge with
             the live pytest run.
        """
        target = snapshot.get("solution.py", "")
        checks = []
        accepted = False

        for pattern in self.correct_patterns:
            found = pattern in target
            checks.append(
                f"Pattern '{pattern}' {'✓ found' if found else '✗ missing'} in solution.py"
            )
            if found:
                accepted = True

        return {
            "accepted": accepted,
            "checks": checks,
        }
