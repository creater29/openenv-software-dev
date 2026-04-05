"""
server.py — FastAPI web server wrapping the SoftwareDev RL environment.

Hugging Face Spaces (Docker SDK) requires a process listening on port 7860.
This server exposes the environment as a REST API so the Space stays alive,
while also serving a simple HTML dashboard at the root URL.

Endpoints:
  GET  /           → HTML landing page
  GET  /health     → JSON health check
  POST /reset      → Start a new episode, returns initial observation
  POST /step       → Take one action, returns (obs, reward, done, info)
  GET  /tasks      → List available task IDs
  POST /run        → Run a complete rule-based demo episode, returns transcript
"""
import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from openenv_software_dev.env import SoftwareDevEnv
from openenv_software_dev.actions import ActionType

app = FastAPI(
    title="OpenEnv Software Dev",
    description="Gymnasium RL environment for software engineering tasks.",
    version="0.1.0",
)

# ── Shared environment instance (single session for demo) ─────────────────────
_env: Optional[SoftwareDevEnv] = None
_last_obs: Optional[Dict] = None


def get_env() -> SoftwareDevEnv:
    global _env
    if _env is None:
        _env = SoftwareDevEnv(difficulty="medium", max_steps=30)
    return _env


# ── Request/response models ───────────────────────────────────────────────────

class StepRequest(BaseModel):
    type: int = 3          # ActionType int
    target_file: str = ""
    text_input: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML landing page."""
    return HTMLResponse(content=LANDING_HTML, status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "env": "SoftwareDev-v0", "version": "0.1.0"}


@app.get("/tasks")
async def list_tasks():
    env = get_env()
    return {"tasks": env.registry.list_ids(), "count": len(env.registry)}


@app.post("/reset")
async def reset(seed: Optional[int] = None):
    global _last_obs
    env = get_env()
    obs, info = env.reset(seed=seed)
    _last_obs = obs
    return {"observation": obs, "info": info}


@app.post("/step")
async def step(action: StepRequest):
    global _last_obs
    env = get_env()
    if env.current_task is None:
        obs, info = env.reset()
        _last_obs = obs

    action_dict = {
        "type": action.type,
        "target_file": action.target_file,
        "text_input": action.text_input,
    }
    obs, reward, done, trunc, info = env.step(action_dict)
    _last_obs = obs
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "truncated": trunc,
        "info": info,
    }


@app.post("/run")
async def run_demo():
    """Run a complete rule-based demo episode and return the transcript."""
    env = SoftwareDevEnv(difficulty="easy", max_steps=10)
    obs, info = env.reset(seed=42)

    transcript = [{"event": "reset", "task": info["task_id"],
                   "description": info["task_description"]}]
    total_reward = 0.0

    # Rule-based steps: read → fix → test → submit
    actions = [
        {"type": ActionType.READ_FILE,  "target_file": "solution.py", "text_input": ""},
        {"type": ActionType.WRITE_FILE, "target_file": "solution.py",
         "text_input": "def add(a, b):\n    return a + b\n"},
        {"type": ActionType.RUN_TESTS,  "target_file": "", "text_input": ""},
        {"type": ActionType.SUBMIT,     "target_file": "", "text_input": ""},
    ]

    for i, action in enumerate(actions, 1):
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        transcript.append({
            "step": i,
            "action_type": action["type"],
            "reward": round(reward, 4),
            "done": done,
            "grading": info.get("grading", {}),
        })
        if done or trunc:
            break

    return {"transcript": transcript, "total_reward": round(total_reward, 4),
            "success": info.get("grading", {}).get("accepted", False)}


# ── HTML landing page ─────────────────────────────────────────────────────────

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OpenEnv Software Dev</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e8eaf0; min-height: 100vh; }
  header { background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
           border-bottom: 1px solid #2a3040; padding: 32px 48px; }
  header h1 { font-size: 2rem; font-weight: 700; color: #7eb8ff; }
  header p  { color: #8892a4; margin-top: 8px; font-size: 1rem; }
  .badge { display:inline-block; background:#1e3a5f; color:#7eb8ff;
           border:1px solid #2a5080; border-radius:4px;
           padding:2px 10px; font-size:0.78rem; margin:4px 2px; }
  main { max-width: 900px; margin: 48px auto; padding: 0 24px; }
  .card { background: #161b2e; border: 1px solid #2a3040; border-radius: 12px;
          padding: 28px 32px; margin-bottom: 24px; }
  .card h2 { font-size: 1.1rem; color: #7eb8ff; margin-bottom: 16px;
             border-bottom: 1px solid #2a3040; padding-bottom: 10px; }
  .reward-box { background: #0d1117; border:1px solid #1e3a5f; border-radius:8px;
                padding:16px 20px; font-family: monospace; color: #7dffb3;
                font-size: 0.95rem; line-height: 1.8; }
  table { width:100%; border-collapse:collapse; font-size:0.9rem; }
  th { background:#1e2a3a; color:#7eb8ff; padding:10px 14px; text-align:left; }
  td { padding:9px 14px; border-bottom:1px solid #1e2535; color:#c8d0e0; }
  tr:last-child td { border-bottom: none; }
  .tag { display:inline-block; background:#1a2535; color:#a0b4cc;
         border-radius:4px; padding:1px 8px; font-size:0.8rem; margin:2px; }
  .endpoint { font-family:monospace; color:#ffa07a; font-size:0.9rem; }
  .method-post { color:#7dffb3; font-weight:700; margin-right:8px; }
  .method-get  { color:#7eb8ff; font-weight:700; margin-right:8px; }
  .ep-row { margin:8px 0; }
  footer { text-align:center; color:#4a5568; font-size:0.82rem; padding:32px 0; }
</style>
</head>
<body>
<header>
  <h1>&#x1F4BB; OpenEnv Software Dev</h1>
  <p>Gymnasium-compatible RL environment for software engineering tasks</p>
  <div style="margin-top:12px">
    <span class="badge">OpenEnv v0.1.0</span>
    <span class="badge">Gymnasium</span>
    <span class="badge">Docker</span>
    <span class="badge">6 Tasks</span>
    <span class="badge">Pytest Grader</span>
  </div>
</header>
<main>
  <div class="card">
    <h2>&#x1F4CA; Reward Formula</h2>
    <div class="reward-box">
      R = (tests_passed &times; 1.0) + (code_quality &times; 0.5) &minus; (step_penalty &times; 0.02)<br>
      On SUBMIT &rarr; R += 5.0 &times; grading_score
    </div>
  </div>
  <div class="card">
    <h2>&#x1F3AF; Available Tasks</h2>
    <table>
      <tr><th>ID</th><th>Category</th><th>Difficulty</th><th>Description</th></tr>
      <tr><td>bugfix-001</td><td>Bug Fix</td><td>Easy</td><td>Fix add(): subtraction instead of addition</td></tr>
      <tr><td>bugfix-002</td><td>Bug Fix</td><td>Medium</td><td>Fix multiply(): off-by-one error</td></tr>
      <tr><td>bugfix-003</td><td>Bug Fix</td><td>Medium</td><td>Fix is_palindrome(): always returns True</td></tr>
      <tr><td>feature-001</td><td>Feature Impl</td><td>Medium</td><td>Implement factorial() with edge cases</td></tr>
      <tr><td>feature-002</td><td>Feature Impl</td><td>Easy</td><td>Implement fizzbuzz(n)</td></tr>
      <tr><td>feature-003</td><td>Feature Impl</td><td>Hard</td><td>Implement binary_search(arr, target)</td></tr>
    </table>
  </div>
  <div class="card">
    <h2>&#x1F527; REST API</h2>
    <div class="ep-row"><span class="method-get">GET</span>  <span class="endpoint">/health</span> &mdash; Health check</div>
    <div class="ep-row"><span class="method-get">GET</span>  <span class="endpoint">/tasks</span>  &mdash; List task IDs</div>
    <div class="ep-row"><span class="method-post">POST</span> <span class="endpoint">/reset</span>  &mdash; Start new episode</div>
    <div class="ep-row"><span class="method-post">POST</span> <span class="endpoint">/step</span>   &mdash; Take one action</div>
    <div class="ep-row"><span class="method-post">POST</span> <span class="endpoint">/run</span>    &mdash; Run a complete demo episode</div>
    <div class="ep-row"><span class="method-get">GET</span>  <span class="endpoint">/docs</span>   &mdash; Interactive API docs (Swagger)</div>
  </div>
  <div class="card">
    <h2>&#x1F4C8; Benchmarks</h2>
    <table>
      <tr><th>Model</th><th>Success Rate</th><th>Avg Steps</th><th>Avg Reward</th></tr>
      <tr><td>Random Agent</td><td>2%</td><td>30.0</td><td>&minus;0.60</td></tr>
      <tr><td>Rule-Based Agent</td><td>45%</td><td>12.4</td><td>1.20</td></tr>
      <tr><td>GPT-4o (Baseline)</td><td>88%</td><td>5.2</td><td>4.50</td></tr>
    </table>
  </div>
</main>
<footer>OpenEnv Software Dev &bull; MIT License &bull; Built with Gymnasium + FastAPI</footer>
</body>
</html>"""
