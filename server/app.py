"""
server/app.py — OpenEnv validator entry point.

The validator explicitly checks for the existence of server/app.py and
expects to find a FastAPI `app` object here. We re-export everything from
server.main so both module paths resolve to the same application, keeping
the Dockerfile CMD (which references server.main:app) fully intact.
"""

from server.main import app  # noqa: F401  re-export for validator

import uvicorn


def main() -> None:
    """Console-script entry point declared in [project.scripts].

    Running `serve` (after `pip install -e .`) will invoke this function,
    which starts the uvicorn server on 0.0.0.0:7860 — the same host/port
    used by the Dockerfile CMD.
    """
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
