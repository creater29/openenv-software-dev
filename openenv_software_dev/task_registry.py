"""
Task Registry — loads, stores, and samples tasks for training episodes.
"""
import random
from typing import List, Optional
from .tasks.bug_fix import BugFixTask
from .tasks.feature_impl import FeatureImplTask


class TaskRegistry:
    """
    Central repository of all available tasks.
    Supports difficulty-based filtering and reproducible sampling via seed.
    """

    def __init__(self):
        self._tasks = []
        self._rng = random.Random()

    def load_defaults(self, difficulty: str = "medium") -> None:
        """Populate the registry with built-in tasks filtered by difficulty."""
        all_tasks = _build_default_tasks()
        self._tasks = [t for t in all_tasks if t.difficulty == difficulty or difficulty == "all"]
        if not self._tasks:
            # Fallback: use everything
            self._tasks = all_tasks

    def register(self, task) -> None:
        self._tasks.append(task)

    def sample(self, seed: Optional[int] = None):
        """Return a randomly chosen task."""
        if seed is not None:
            self._rng.seed(seed)
        if not self._tasks:
            raise RuntimeError("TaskRegistry is empty. Call load_defaults() first.")
        return self._rng.choice(self._tasks)

    def list_ids(self) -> List[str]:
        return [t.task_id for t in self._tasks]

    def __len__(self) -> int:
        return len(self._tasks)


# ── Default task catalogue ────────────────────────────────────────────────────

def _build_default_tasks():
    return [
        # ── Bug-fix tasks ─────────────────────────────────────────────
        BugFixTask(
            task_id="bugfix-001",
            description=(
                "The `add` function in solution.py is broken — it subtracts instead of adds. "
                "Fix the bug so all tests pass."
            ),
            difficulty="easy",
            hints=["Look at the operator used in the return statement."],
        ),
        BugFixTask(
            task_id="bugfix-002",
            description=(
                "The `multiply` function has an off-by-one error. "
                "It multiplies by (b-1) instead of b. Fix it."
            ),
            difficulty="medium",
            starter_files={
                "solution.py": "def multiply(a, b):\n    return a * (b - 1)\n",
                "test_solution.py": (
                    "from solution import multiply\n"
                    "def test_multiply():\n"
                    "    assert multiply(3, 4) == 12\n"
                    "    assert multiply(5, 0) == 0\n"
                ),
            },
            hints=["The loop/expression is almost right — check the operand."],
        ),
        BugFixTask(
            task_id="bugfix-003",
            description=(
                "The `is_palindrome` function returns True for every string. "
                "Implement it correctly."
            ),
            difficulty="medium",
            starter_files={
                "solution.py": "def is_palindrome(s):\n    return True\n",
                "test_solution.py": (
                    "from solution import is_palindrome\n"
                    "def test_palindrome():\n"
                    "    assert is_palindrome('racecar') is True\n"
                    "    assert is_palindrome('hello') is False\n"
                ),
            },
            hints=["Compare the string with its reverse."],
        ),
        # ── Feature-implementation tasks ──────────────────────────────
        FeatureImplTask(
            task_id="feature-001",
            description=(
                "Implement a `factorial` function in solution.py. "
                "It must handle 0! = 1 and raise ValueError for negative inputs."
            ),
            difficulty="medium",
            hints=[
                "Use recursion or a loop.",
                "Don't forget the base case.",
            ],
        ),
        FeatureImplTask(
            task_id="feature-002",
            description=(
                "Implement a `fizzbuzz(n)` function that returns a list of strings "
                "from 1 to n: 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, "
                "otherwise the number as a string."
            ),
            difficulty="easy",
            starter_files={
                "solution.py": "def fizzbuzz(n):\n    pass\n",
                "test_solution.py": (
                    "from solution import fizzbuzz\n"
                    "def test_fizzbuzz():\n"
                    "    result = fizzbuzz(15)\n"
                    "    assert result[2] == 'Fizz'\n"
                    "    assert result[4] == 'Buzz'\n"
                    "    assert result[14] == 'FizzBuzz'\n"
                ),
            },
            hints=["Check divisibility with the % operator."],
        ),
        FeatureImplTask(
            task_id="feature-003",
            description=(
                "Implement `binary_search(arr, target)` in solution.py. "
                "Return the index of target in the sorted list arr, or -1 if not found."
            ),
            difficulty="hard",
            starter_files={
                "solution.py": "def binary_search(arr, target):\n    pass\n",
                "test_solution.py": (
                    "from solution import binary_search\n"
                    "def test_binary_search():\n"
                    "    assert binary_search([1,3,5,7,9], 5) == 2\n"
                    "    assert binary_search([1,3,5,7,9], 6) == -1\n"
                    "    assert binary_search([], 1) == -1\n"
                ),
            },
            hints=["Track lo and hi pointers; mid = (lo + hi) // 2."],
        ),
    ]
