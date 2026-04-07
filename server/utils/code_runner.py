from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, TypedDict


class SandboxResult(TypedDict):
    passed: int
    failed: int
    total: int
    output: str
    returncode: int


def _parse_pytest_counts(output: str) -> Dict[str, int]:
    passed = 0
    failed = 0

    passed_match = re.search(r"(\d+)\s+passed", output)
    failed_match = re.search(r"(\d+)\s+failed", output)

    if passed_match:
        passed = int(passed_match.group(1))
    if failed_match:
        failed = int(failed_match.group(1))

    total = passed + failed
    if total == 0 and "error" in output.lower():
        total = 1
        failed = 1

    return {"passed": passed, "failed": failed, "total": total}


def _to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="ignore")
    if isinstance(value, memoryview):
        return value.tobytes().decode("utf-8", errors="ignore")
    return str(value)


def run_pytest_in_sandbox(files: Dict[str, str], tests: Dict[str, str], timeout_seconds: int = 8) -> SandboxResult:
    """Run tests in a temporary directory with deterministic pytest settings."""

    with tempfile.TemporaryDirectory(prefix="swe_sim_") as tmp:
        root = Path(tmp)

        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        for name, content in tests.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--maxfail=20"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
            output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            counts = _parse_pytest_counts(output)
            return {
                "passed": counts["passed"],
                "failed": counts["failed"],
                "total": counts["total"],
                "output": output.strip(),
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            stdout = _to_text(exc.stdout)
            stderr = _to_text(exc.stderr)
            out = (stdout + "\n" + stderr).strip()
            return {
                "passed": 0,
                "failed": 1,
                "total": 1,
                "output": f"Test run timed out after {timeout_seconds}s\n{out}".strip(),
                "returncode": 124,
            }
