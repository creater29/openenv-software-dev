"""
Tests for the grader subsystem.
Run with:  pytest tests/test_graders.py -v
"""
import pytest
from openenv_software_dev.graders.programmatic import ProgrammaticGrader
from openenv_software_dev.graders.composite import CompositeGrader
from openenv_software_dev.sandbox.filesystem import VirtualFilesystem
from openenv_software_dev.sandbox.executor import SandboxedExecutor
from openenv_software_dev.tasks.bug_fix import BugFixTask
from openenv_software_dev.tasks.feature_impl import FeatureImplTask


@pytest.fixture
def bug_task():
    return BugFixTask(
        task_id="test-bugfix",
        description="Fix the add function.",
        difficulty="easy",
    )


@pytest.fixture
def executor():
    return SandboxedExecutor(timeout=10)


def _make_vfs(files: dict) -> VirtualFilesystem:
    vfs = VirtualFilesystem()
    for path, content in files.items():
        vfs.write(path, content)
    return vfs


def test_programmatic_grader_broken_code(bug_task, executor):
    """Broken code should yield a low score."""
    vfs = _make_vfs({
        "solution.py": "def add(a, b):\n    return a - b\n",
        "test_solution.py": (
            "from solution import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
        ),
    })
    grader = ProgrammaticGrader()
    result = grader.grade(bug_task, vfs, executor)
    assert result["score"] < 0.5
    assert result["accepted"] is False


def test_programmatic_grader_fixed_code(bug_task, executor):
    """Fixed code should yield a high score."""
    vfs = _make_vfs({
        "solution.py": "def add(a, b):\n    return a + b\n",
        "test_solution.py": (
            "from solution import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
        ),
    })
    grader = ProgrammaticGrader()
    result = grader.grade(bug_task, vfs, executor)
    assert result["accepted"] is True
    assert result["tests_passed"] == 1.0


def test_composite_grader_no_llm(bug_task, executor):
    """CompositeGrader with LLM disabled should still return a valid result."""
    vfs = _make_vfs({
        "solution.py": "def add(a, b):\n    return a + b\n",
        "test_solution.py": (
            "from solution import add\n"
            "def test_add(): assert add(1, 2) == 3\n"
        ),
    })
    grader = CompositeGrader(enable_llm=False)
    result = grader.grade(bug_task, vfs, executor)
    assert 0.0 <= result["score"] <= 1.0
    assert "tests_passed" in result
    assert result["llm"] == {}


def test_feature_impl_acceptance_stub():
    """A stub solution should NOT pass the static acceptance check."""
    task = FeatureImplTask(task_id="test-feature", description="impl factorial", difficulty="medium")
    vfs = _make_vfs(task.starter_files)
    result = task.acceptance_check(vfs.snapshot())
    assert result["accepted"] is False


def test_feature_impl_acceptance_implemented():
    """An implemented solution should pass the static check."""
    task = FeatureImplTask(task_id="test-feature", description="impl factorial", difficulty="medium")
    vfs = _make_vfs({
        "solution.py": (
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
    })
    result = task.acceptance_check(vfs.snapshot())
    assert result["accepted"] is True
