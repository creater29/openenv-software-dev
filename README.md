---
title: SWE-Sim OpenEnv Software Engineering Simulator
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - software-engineering
  - debugging
  - reinforcement-learning
  - rl
  - multi-step
license: mit
---

# SWE-Sim: OpenEnv Software Engineering Simulator

> An RL environment where agents debug, fix, and improve real-world codebases.  
> Inspired by real-world benchmarks for autonomous software engineering agents.

## 🧠 What Makes This Unique
- Simulates real software engineering workflows (not toy problems)
- Supports multi-step, stateful debugging across multiple files
- Shaped reward signal encourages iterative improvement and efficiency
- Includes hidden test evaluation to prevent reward hacking
- Difficulty scaling: easy / medium / hard per task
- Designed for training and evaluating autonomous coding agents

## 🎯 Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| fix_broken_api | Easy | Fix a broken production API route |
| resolve_ci_pipeline | Medium | Resolve multi-file failing CI pipeline |
| debug_hidden_state | Hard | Debug a 3-file system with hidden tests |

## 📐 Action & Observation Spaces
Observation (`ObservationModel`):
- `task: str` - Active task name.
- `difficulty: str` - `easy | medium | hard`.
- `step: int` - Current step counter.
- `max_steps: int` - Episode cap.
- `files: dict[str, str] | null` - Multi-file code state.
- `file: str | null` - Single-file task filename.
- `code: str | null` - Single-file task code body.
- `error_log: str | null` - Latest traceback or pytest output.
- `test_results: str | null` - Human-readable pass/fail summary.
- `visible_tests: str | null` - Visible test status for hidden-state task.
- `hidden_tests: str | null` - Hidden test message shown to the agent.
- `last_reward: float` - Previous step reward.
- `metadata: dict[str, Any]` - Allowed actions and reward weights.

Action (`ActionModel`):
- `action: str` - `inspect | patch | run_tests | submit` (task-dependent).
- `filename: str | null` - Target file for inspect/patch.
- `code: str | null` - Replacement code for patch/submit.

Reward (`RewardModel`):
- `reward: float` - Final shaped reward in `[0, 1]`.
- `tests_passed_ratio: float` - Current pass ratio.
- `improvement_over_last_step: float` - Improvement signal.
- `step_penalty: float` - Step efficiency penalty.
- `destructive_action_penalty: float` - Penalty for harmful rewrites.
- `components: dict[str, float]` - Reward term decomposition.

## 🏆 Reward Design
Reward is shaped to encourage iterative debugging and efficiency.

Shared template:
`reward = (tests_passed_ratio * W_pass) + (improvement_over_last_step * W_improve) - (steps_taken * W_step_penalty) - destructive_action_penalty`

Task formulas and weights:
- `fix_broken_api`:
  `reward = (ratio * 0.75) + (improvement * 0.25) - (steps * 0.03) - destructive_penalty`
- `resolve_ci_pipeline`:
  `reward = (ratio * 1.00) + (improvement * 0.30) - (steps * 0.05) - destructive_penalty`
- `debug_hidden_state`:
  `reward = (0.5 * visible_ratio) + (0.5 * hidden_ratio) + (0.2 * improvement) - (steps * 0.02) - destructive_penalty`
  and then `reward = clamp(reward, 0.0, 1.0)`

## 🚀 Example Interaction
POST `/reset` → `{"task": "debug_hidden_state", "difficulty": "hard"}`

Response:
```json
{
  "observation": {
    "task": "debug_hidden_state",
    "difficulty": "hard",
    "step": 0,
    "max_steps": 14,
    "files": {
      "api.py": "from db import thresholded_values\\n...",
      "db.py": "from config import get_config, reset_runtime\\n...",
      "config.py": "DEFAULT_CONFIG = {'threshold': 10, 'window': 2}\\n..."
    },
    "error_log": "TypeError: NoneType object is not subscriptable (line 42)",
    "visible_tests": "1 passing, 3 failing",
    "hidden_tests": "[HIDDEN - evaluated at submit]",
    "last_reward": 0.0,
    "metadata": {
      "allowed_actions": ["inspect", "patch", "run_tests", "submit"]
    }
  },
  "reward": {
    "reward": 0.0,
    "tests_passed_ratio": 0.0,
    "improvement_over_last_step": 0.0,
    "step_penalty": 0.0,
    "destructive_action_penalty": 0.0,
    "components": {"reset": 0.0}
  },
  "done": false,
  "info": {"message": "Environment reset"}
}
```

POST `/step` → `{"action": "inspect", "filename": "api.py"}`
POST `/step` → `{"action": "patch", "filename": "config.py", "code": "..."}`
POST `/step` → `{"action": "run_tests"}`
POST `/step` → `{"action": "submit"}`

Example `/step` response:
```json
{
  "observation": {
    "task": "debug_hidden_state",
    "difficulty": "hard",
    "step": 4,
    "max_steps": 14,
    "visible_tests": "4 passing, 0 failing",
    "hidden_tests": "[HIDDEN - evaluated at submit]",
    "last_reward": 0.94
  },
  "reward": {
    "reward": 0.94,
    "tests_passed_ratio": 1.0,
    "improvement_over_last_step": 0.5,
    "step_penalty": 0.08,
    "destructive_action_penalty": 0.0,
    "components": {
      "visible": 0.5,
      "hidden": 0.5,
      "improve": 0.1,
      "step": -0.08,
      "destructive": 0.0
    }
  },
  "done": true,
  "info": {
    "action": "submit",
    "score": 1.0,
    "hidden_passed": 3,
    "hidden_total": 3
  }
}
```

## 📊 Baseline Scores

| task | model | score | steps | notes |
|------|-------|-------|-------|-------|
| fix_broken_api | GPT-4 class | 0.93 | 2 | typically inspects once then submits full fix |
| resolve_ci_pipeline | GPT-4 class | 0.88 | 4 | usually patches `utils.py`, then reruns tests |
| debug_hidden_state | GPT-4 class | 0.81 | 6 | hidden tests require robust state-reset reasoning |

## ⚙️ Setup & Usage
Docker:
```bash
docker build -t swe-sim .
docker run -p 7860:7860 swe-sim
```

Validate OpenEnv metadata:
```bash
openenv validate openenv.yaml
```

Run baseline inference:
```bash
export OPENENV_URL="http://localhost:7860"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="<token>"
python inference.py
```

API endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`

## 🔬 Research Context
"This environment is inspired by real-world RL benchmarks for autonomous software
agents, filling a gap between toy coding tasks and full SWE-Bench complexity."
