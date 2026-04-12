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

## What Makes This Unique
- Simulates real software engineering workflows (not toy problems)
- Supports multi-step, stateful debugging across multiple files
- Shaped reward signal encourages iterative improvement and efficiency
- Includes hidden test evaluation to prevent reward hacking
- Difficulty scaling: easy / medium / hard per task
- Designed for training and evaluating autonomous coding agents

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| fix_broken_api | Easy | Fix a broken production API route |
| resolve_ci_pipeline | Medium | Resolve multi-file failing CI pipeline |
| debug_hidden_state | Hard | Debug a 3-file system with hidden tests |

## Action and Observation Spaces

Observation (`ObservationModel`):
- `task: str` - Active task name.
- `difficulty: str` - easy, medium, or hard.
- `step: int` - Current step counter.
- `max_steps: int` - Episode cap.
- `files: dict[str, str] | null` - Multi-file code state.
- `file: str | null` - Single-file task filename.
- `code: str | null` - Single-file task code body.
- `error_log: str | null` - Latest traceback or pytest output.
- `test_results: str | null` - Human-readable pass/fail summary.
- `visible_tests: str | null` - Visible test status for hidden-state task.
- `hidden_tests: str | null` - Hidden test message shown to the agent.
- `last_reward: float` - Previous step reward, strictly in (0, 1).
- `metadata: dict` - Allowed actions and reward weights.

Action (`ActionModel`):
- `action: str` - inspect, patch, run_tests, or submit (task-dependent).
- `filename: str | null` - Target file for inspect/patch.
- `code: str | null` - Replacement code for patch/submit.

Reward (`RewardModel`): all float fields are strictly in the open interval (0, 1).
- `reward: float` - Final shaped reward.
- `tests_passed_ratio: float` - Current pass ratio.
- `improvement_over_last_step: float` - Improvement signal.
- `step_penalty: float` - Step efficiency penalty.
- `destructive_action_penalty: float` - Penalty for harmful rewrites.
- `components: dict[str, float]` - Reward term decomposition.

## Reward Design

Reward is shaped to encourage iterative debugging and efficiency.
All raw values are clamped to the open interval (0.001, 0.999) before being returned.

Shared template:
`reward = (tests_passed_ratio * W_pass) + (improvement * W_improve) - (steps * W_step_penalty) - destructive_penalty`

Task weights:
- `fix_broken_api`: W_pass=0.75, W_improve=0.25, W_step=0.03
- `resolve_ci_pipeline`: W_pass=1.00, W_improve=0.30, W_step=0.05
- `debug_hidden_state`: 0.5*visible + 0.5*hidden + 0.2*improve - 0.02*steps

## Example Interaction

POST `/reset` with `{"task": "debug_hidden_state", "difficulty": "hard"}` returns a
`StepResponse` with all reward fields initialized to 0.001 and the initial
observation containing the broken source files and error log.

POST `/step` with `{"action": "submit"}` returns shaped reward based on
visible and hidden test outcomes. All float fields in the response are
guaranteed to be strictly between 0 and 1 (open interval).

## Baseline Scores

Scores are shaped rewards clamped to the open interval (0.001, 0.999). The formula
charges a per-step penalty, so even a perfect fix does not reach 1.0.

| Task | Agent | Reward | Steps | Notes |
|------|-------|--------|-------|-------|
| fix_broken_api | Heuristic | 0.939 | 2 | inspect → submit; all 3 hidden tests pass |
| resolve_ci_pipeline | Heuristic | 0.849 | 3 | patch utils.py → run_tests → submit |
| debug_hidden_state | Heuristic | 0.939 | 3 | patch config.py → run_tests → submit |

Formula derivation for `fix_broken_api` (2 steps, W_pass=0.75, W_improve=0.25, W_step=0.03):
`reward = (1.0×0.75) + (1.0×0.25) − (2×0.03) − 0.001 = 0.939`

For `resolve_ci_pipeline` (3 steps, W_pass=1.0, W_improve=0.3, W_step=0.05):
improvement=0 at submit because run_tests already raised ratio to 1.0:
`reward = (1.0×1.0) + (0.0×0.3) − (3×0.05) − 0.001 = 0.849`

For `debug_hidden_state` (3 steps, 0.5×visible + 0.5×hidden, W_improve=0.2, W_step=0.02):
`reward = (0.5×1.0 + 0.5×1.0) + (0.0×0.2) − (3×0.02) − 0.001 = 0.939`

## Setup and Usage

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

API endpoints: `POST /reset`, `POST /step`, `GET /state`

## Research Context

This environment is inspired by real-world RL benchmarks for autonomous software
agents, filling a gap between toy coding tasks and full SWE-Bench complexity.
