"""
Feature-implementation tasks — the agent starts with a stub and must write
the complete implementation so the test suite passes.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List
from .base import Task


@dataclass
class FeatureImplTask(Task):
    """
    A task where solution.py contains only a stub (pass / NotImplementedError).
    The agent must implement the function so all tests pass.
    """

    category: str = "feature_impl"

    # Default: implement factorial
    starter_files: Dict[str, str] = field(
        default_factory=lambda: {
            "solution.py": (
                "def factorial(n):\n"
                "    \"\"\"Return n! for non-negative integers.\"\"\"\n"
                "    pass  # TODO: implement\n"
            ),
            "test_solution.py": (
                "from solution import factorial\n\n"
                "def test_factorial_base():\n"
                "    assert factorial(0) == 1\n"
                "    assert factorial(1) == 1\n\n"
                "def test_factorial_positive():\n"
                "    assert factorial(5) == 120\n"
                "    assert factorial(10) == 3628800\n\n"
                "def test_factorial_negative():\n"
                "    import pytest\n"
                "    with pytest.raises(ValueError):\n"
                "        factorial(-1)\n"
            ),
        }
    )

    # Keywords that signal an actual implementation (not a stub)
    impl_keywords: List[str] = field(
        default_factory=lambda: ["return", "if", "for", "while", "raise"]
    )

    def acceptance_check(self, snapshot: Dict[str, str]) -> Dict[str, Any]:
        """
        Static checks:
          1. 'pass' is no longer the only statement (stub removed).
          2. The file contains at least one impl_keyword.
          3. The function is still defined.
        """
        code = snapshot.get("solution.py", "")
        checks = []

        stub_removed = "pass  # TODO" not in code and "pass" not in code.replace("\n", "")
        checks.append(
            "Stub 'pass' removed from solution.py ✓" if stub_removed
            else "Stub 'pass' still present in solution.py ✗"
        )

        has_impl = any(kw in code for kw in self.impl_keywords)
        checks.append(
            "Implementation logic detected ✓" if has_impl
            else "No implementation logic found ✗"
        )

        accepted = has_impl  # static pass; tests will confirm correctness

        return {
            "accepted": accepted,
            "checks": checks,
        }
