"""
Action definitions for the SoftwareDev environment.
Defines the discrete action types an agent can take.
"""
from enum import IntEnum
from typing import Optional


class ActionType(IntEnum):
    READ_FILE = 0
    WRITE_FILE = 1
    RUN_TESTS = 2
    SUBMIT = 3
    LIST_FILES = 4


class ActionRegistry:
    """Registry for all valid actions in the environment."""

    ALL_ACTIONS = [
        {
            "id": ActionType.READ_FILE,
            "name": "read_file",
            "description": "Read the contents of a file from the virtual filesystem.",
            "requires": ["target_file"],
        },
        {
            "id": ActionType.WRITE_FILE,
            "name": "write_file",
            "description": "Write content to a file in the virtual filesystem.",
            "requires": ["target_file", "text_input"],
        },
        {
            "id": ActionType.RUN_TESTS,
            "name": "run_tests",
            "description": "Run the test suite against the current code.",
            "requires": [],
        },
        {
            "id": ActionType.SUBMIT,
            "name": "submit",
            "description": "Submit the current solution for final grading.",
            "requires": [],
        },
        {
            "id": ActionType.LIST_FILES,
            "name": "list_files",
            "description": "List all files currently in the virtual filesystem.",
            "requires": [],
        },
    ]

    @classmethod
    def describe(cls) -> str:
        lines = []
        for action in cls.ALL_ACTIONS:
            lines.append(
                f"  {action['id']}: {action['name']} — {action['description']}"
            )
        return "\n".join(lines)
