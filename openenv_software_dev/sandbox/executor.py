"""
SandboxedExecutor — runs agent-written code safely in a temporary directory.

SECURITY MODEL
--------------
Agent-provided code never runs on the live host filesystem.  Instead:
  1. The VirtualFilesystem is materialised into a fresh tempdir.
  2. pytest is invoked as a subprocess with a hard timeout.
  3. The tempdir is deleted automatically when the context manager exits.

This keeps the agent's edits isolated from the real project even if they
write malicious file paths (because tempfile.TemporaryDirectory is scoped).
"""
import json
import os
import subprocess
import tempfile
from typing import Any, Dict

from .filesystem import VirtualFilesystem


class SandboxedExecutor:
    """
    Materialises the VFS to disk and runs pytest, returning structured results.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout  # seconds before tests are killed

    def run_tests(self, vfs: VirtualFilesystem) -> Dict[str, Any]:
        """
        Run the test suite against the current VFS state.

        Returns a dict with:
          status      : "success" | "failure" | "timeout" | "error"
          output      : raw pytest stdout (truncated to 4000 chars)
          tests_passed: fraction of tests that passed (0.0–1.0)
          passed      : number of tests passed
          failed      : number of tests failed
          total       : total number of tests collected
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # ── 1. Materialise the VFS ────────────────────────────────
            for path, content in vfs.snapshot().items():
                full_path = os.path.join(tmpdir, path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as fh:
                    fh.write(content)

            # ── 2. Run pytest ─────────────────────────────────────────
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", "--tb=short", "-q"],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=self.timeout,
                )
                raw_output = (result.stdout + result.stderr).decode("utf-8", errors="replace")

                # ── 3. Parse outcome ──────────────────────────────────
                passed, failed, total = _parse_pytest_summary(raw_output)
                fraction = (passed / total) if total > 0 else 0.0
                status = "success" if failed == 0 and total > 0 else "failure"

                return {
                    "status": status,
                    "output": raw_output[:4000],   # cap to avoid huge obs
                    "tests_passed": round(fraction, 4),
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                }

            except subprocess.TimeoutExpired:
                return {
                    "status": "timeout",
                    "output": f"Tests timed out after {self.timeout}s.",
                    "tests_passed": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "total": 0,
                }
            except FileNotFoundError:
                # pytest not installed in this environment
                return {
                    "status": "error",
                    "output": "pytest not found — install it with: pip install pytest",
                    "tests_passed": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "total": 0,
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "output": str(exc),
                    "tests_passed": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "total": 0,
                }


def _parse_pytest_summary(output: str):
    """
    Extract pass/fail counts from the last summary line of pytest output.

    Pytest typically ends with a line like:
      '1 passed, 2 failed in 0.42s'
    or just:
      '3 passed in 0.10s'
    """
    passed = failed = 0
    for line in reversed(output.splitlines()):
        line = line.strip()
        if "passed" in line or "failed" in line or "error" in line:
            # Extract numbers preceding the keywords
            import re
            p = re.search(r"(\d+) passed", line)
            f = re.search(r"(\d+) failed", line)
            e = re.search(r"(\d+) error", line)
            passed = int(p.group(1)) if p else 0
            failed = int(f.group(1)) if f else 0
            failed += int(e.group(1)) if e else 0
            break
    total = passed + failed
    return passed, failed, total
