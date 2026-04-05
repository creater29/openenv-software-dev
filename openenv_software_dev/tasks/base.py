"""
Abstract base class for all tasks in the SoftwareDev environment.

Every concrete task (bug fix, feature implementation, code review…) inherits
from Task and must implement `acceptance_check`.  The dataclass pattern keeps
task definitions short and self-documenting.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Task:
    """
    Base task definition.

    Attributes
    ----------
    task_id         Unique string identifier (e.g. 'bugfix-001').
    description     Natural-language problem statement shown to the agent.
    difficulty      One of 'easy', 'medium', 'hard'.
    starter_files   Dict mapping filename → initial file content.
    hints           Optional list of hints the agent may read.
    max_score       Maximum achievable score (used for normalisation).
    """

    task_id:       str             = "task-000"
    description:   str             = "No description provided."
    difficulty:    str             = "medium"
    category:      str             = "generic"
    starter_files: Dict[str, str]  = field(default_factory=dict)
    hints:         List[str]       = field(default_factory=list)
    max_score:     float           = 1.0

    def acceptance_check(self, snapshot: Dict[str, str]) -> Dict[str, Any]:
        """
        Determine whether the agent's solution is acceptable.

        Subclasses *must* override this.  Return a dict with at least:
          accepted : bool   — did the agent solve the task?
          checks   : list   — human-readable list of criteria and their status.
        """
        raise NotImplementedError(
            f"Task '{self.task_id}' must implement acceptance_check()."
        )


@dataclass
class TaskMetrics:
    """Lightweight result container for a single acceptance_check call."""
    accepted: bool
    checks:   List[str]
    details:  Dict[str, Any] = field(default_factory=dict)
