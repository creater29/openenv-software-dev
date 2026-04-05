---
title: OpenEnv Software Dev
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - reinforcement-learning
  - gymnasium
  - llm-agent
  - software-engineering
  - openenv
---

# 💻 OpenEnv Software Development Environment

A **Gymnasium-compatible Reinforcement Learning environment** for software engineering tasks, built to the [OpenEnv specification](https://huggingface.co/OpenEnv).

An LLM agent reads code, edits files in a virtual filesystem, runs tests via pytest, and receives shaped rewards for solving tasks like bug fixes and feature implementations.

---

## 🚀 Quick Start

```bash
pip install gymnasium
pip install git+https://huggingface.co/spaces/EndlessMarathon/openenv-software-dev.git

# Run the baseline inference script
HF_TOKEN=hf_xxx python inference.py
```

---

## 🛠 How It Works

1. **Observation** — Agent receives a JSON dict with the file tree, full file contents, task description, progress, and last reward.
2. **Action** — Agent sends one of: `READ_FILE`, `WRITE_FILE`, `RUN_TESTS`, `SUBMIT`, `LIST_FILES`.
3. **Reward** — Shaped reward computed every step:

```
R = (tests_passed × 1.0) + (code_quality × 0.5) − (step_penalty × 0.02)
```

On `SUBMIT`, a terminal bonus is added:

```
R += 5.0 × grading_score
```

---

## 📂 Project Structure

```
openenv-software-dev/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── openenv.yaml                  # OpenEnv metadata
├── inference.py                  # Baseline inference script
├── validate-submission.sh
├── configs/
│   ├── default.yaml
│   └── hard_mode.yaml
├── openenv_software_dev/
│   ├── env.py                    # Core Gymnasium environment
│   ├── actions.py                # Action definitions
│   ├── observation.py            # Observation builder
│   ├── reward.py                 # Reward shaping
│   ├── task_registry.py          # Task catalogue
│   ├── registration.py           # Gymnasium registration
│   ├── graders/
│   │   ├── programmatic.py       # Deterministic pytest grader
│   │   ├── llm_grader.py         # Semantic LLM grader
│   │   └── composite.py          # Weighted composite grader
│   ├── tasks/
│   │   ├── base.py               # Abstract task class
│   │   ├── bug_fix.py            # Bug-fix tasks
│   │   └── feature_impl.py       # Feature implementation tasks
│   └── sandbox/
│       ├── filesystem.py         # Virtual in-memory filesystem
│       └── executor.py           # Sandboxed code runner
└── tests/
    ├── test_env.py
    ├── test_graders.py
    └── test_integration.py
```

---

## 🎯 Tasks

| ID | Category | Difficulty | Description |
|---|---|---|---|
| bugfix-001 | Bug Fix | Easy | Fix `add`: subtraction instead of addition |
| bugfix-002 | Bug Fix | Medium | Fix `multiply`: off-by-one in multiplier |
| bugfix-003 | Bug Fix | Medium | Fix `is_palindrome`: always returns True |
| feature-001 | Feature Impl | Medium | Implement `factorial` with edge cases |
| feature-002 | Feature Impl | Easy | Implement `fizzbuzz(n)` |
| feature-003 | Feature Impl | Hard | Implement `binary_search(arr, target)` |

---

## 🌍 Environments

| Gym ID | Difficulty | Max Steps |
|---|---|---|
| `SoftwareDev-easy-v0` | Easy | 15 |
| `SoftwareDev-v0` | Medium | 30 |
| `SoftwareDev-hard-v0` | Hard | 50 |

```python
import gymnasium as gym
env = gym.make("SoftwareDev-v0")
obs, info = env.reset()
```

---

## 📈 Performance Benchmarks

| Model | Success Rate | Avg Steps | Avg Reward |
|:---|:---:|:---:|:---:|
| Random Agent | 2% | 30.0 | −0.60 |
| Rule-Based Agent | 45% | 12.4 | 1.20 |
| GPT-4o (Baseline) | 88% | 5.2 | 4.50 |

---

## 🏗 Action Space

```python
action = {
    "type": 1,                    # ActionType int (0–4)
    "target_file": "solution.py", # filename in virtual FS
    "text_input": "def add(...)", # content for WRITE_FILE
}
obs, reward, done, truncated, info = env.step(action)
```

| Type | Name | Description |
|---|---|---|
| 0 | READ_FILE | Read a file from the virtual FS |
| 1 | WRITE_FILE | Write/overwrite a file |
| 2 | RUN_TESTS | Execute pytest against current code |
| 3 | SUBMIT | Submit for final grading (terminates episode) |
| 4 | LIST_FILES | List all files in the virtual FS |

---

## ✅ Compliance Checklist

- [x] 3+ tasks (BugFix ×3, FeatureImpl ×3)
- [x] Programmatic grader (pytest-based)
- [x] LLM semantic grader (optional, composite)
- [x] Shaped reward with progress + quality + efficiency
- [x] Inference script with `[START]` / `[STEP]` / `[END]` format
- [x] Dockerized
- [x] Hugging Face Space compatible (`openenv.yaml` + metadata)
- [x] Baseline scores in README

---

## 🔧 Local Development

```bash
git clone https://huggingface.co/spaces/EndlessMarathon/openenv-software-dev
cd openenv-software-dev
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Validate submission
bash validate-submission.sh

# Docker
docker compose up --build
```

---

## 📄 License

MIT
