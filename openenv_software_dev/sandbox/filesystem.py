"""
Virtual Filesystem — in-memory key-value store that simulates a real directory.

The VFS is the single source of truth for all files an agent creates or edits.
Because every file is just a dict entry, we can snapshot it cheaply and replay
history without touching the real host filesystem during training.
"""
from typing import Dict, List, Optional


class VirtualFilesystem:
    """
    Lightweight in-memory filesystem for sandboxed code execution.

    All paths are stored as plain strings (e.g. 'solution.py' or 'src/utils.py').
    The class maintains a full history of writes so we can always roll back or
    inspect what changed at each step.
    """

    def __init__(self):
        self._files: Dict[str, str] = {}     # path → content
        self._history: List[Dict[str, str]] = []  # stack of snapshots

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(self, path: str, content: str) -> None:
        """Create or overwrite a file.  Saves the previous state for undo support."""
        # Push current state onto history before mutating
        self._history.append(self._files.copy())
        self._files[path] = content

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self, path: str) -> str:
        """Return file contents, or a clear error message if not found."""
        if path not in self._files:
            return f"Error: File '{path}' not found. Available: {sorted(self._files.keys())}"
        return self._files[path]

    # ── Directory listing ─────────────────────────────────────────────────────

    def list_files(self) -> List[str]:
        """Return a sorted list of all paths currently in the VFS."""
        return sorted(self._files.keys())

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, str]:
        """Return a shallow copy of the current file state (safe for serialisation)."""
        return self._files.copy()

    # ── Undo ─────────────────────────────────────────────────────────────────

    def undo(self) -> bool:
        """Roll back the most recent write.  Returns False if no history exists."""
        if not self._history:
            return False
        self._files = self._history.pop()
        return True

    def __repr__(self) -> str:
        return f"VirtualFilesystem({list(self._files.keys())})"
