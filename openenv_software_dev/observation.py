"""
Observation builder for the SoftwareDev environment.
Converts raw environment state into a structured dict the agent can read.
"""
import json
from typing import Any, Dict


class ObservationBuilder:
    """
    Assembles a rich observation from the current task state, filesystem
    contents, and last-action outcome.  The observation is a plain Python
    dict so it works with both LLM-based and classical RL agents.
    """

    def build(
        self,
        task,
        fs,
        last_action_status: str,
        progress: float,
        last_reward: float,
    ) -> Dict[str, Any]:
        """Return the full observation dict for the current step."""
        snapshot = fs.snapshot()

        return {
            # ── Task context ──────────────────────────────────────────
            "task_id": task.task_id,
            "task_description": task.description,
            "task_category": task.category,
            "hints": task.hints,

            # ── File system state ─────────────────────────────────────
            "file_tree": sorted(snapshot.keys()),
            "files": snapshot,  # full content of every file

            # ── Episode progress ──────────────────────────────────────
            "progress": round(progress, 4),   # 0.0 → 1.0 (steps used)
            "last_action_status": last_action_status,
            "last_reward": round(last_reward, 4),

            # ── Agent guidance ────────────────────────────────────────
            "available_actions": [
                "0: read_file   — read a file (requires target_file)",
                "1: write_file  — write to a file (requires target_file + text_input)",
                "2: run_tests   — execute the test suite",
                "3: submit      — submit for final grading",
                "4: list_files  — list all files in the virtual FS",
            ],
        }

    def to_text(self, obs: Dict[str, Any]) -> str:
        """Render the observation as a human-readable string for LLM prompts."""
        lines = [
            f"TASK [{obs['task_id']}]: {obs['task_description']}",
            f"Category: {obs['task_category']}",
            f"Progress: {obs['progress']*100:.1f}% of steps used",
            "",
            "=== FILE TREE ===",
        ]
        for fname in obs["file_tree"]:
            lines.append(f"  {fname}")
        lines.append("")
        lines.append("=== FILE CONTENTS ===")
        for fname, content in obs["files"].items():
            lines.append(f"--- {fname} ---")
            lines.append(content)
            lines.append("")
        if obs["hints"]:
            lines.append("=== HINTS ===")
            for h in obs["hints"]:
                lines.append(f"  • {h}")
        lines.append("")
        lines.append("=== AVAILABLE ACTIONS ===")
        for a in obs["available_actions"]:
            lines.append(f"  {a}")
        return "\n".join(lines)
