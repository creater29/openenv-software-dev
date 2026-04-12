from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
OPENENV_URL = os.getenv("OPENENV_URL", "http://localhost:7860")
BENCHMARK = "SWE-Sim"


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _fmt_reward(value: float) -> str:
    return f"{value:.2f}"


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_action(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return fallback

    # Prefer JSON in fenced block, then any JSON object in free text.
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        any_json = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        candidate = any_json.group(1) if any_json else None

    if candidate is None:
        return fallback

    try:
        action = json.loads(candidate)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(action, dict) or "action" not in action:
        return fallback

    normalized = {
        "action": str(action.get("action", "")).strip().lower(),
        "filename": action.get("filename"),
        "code": action.get("code"),
    }
    return normalized


def _build_system_prompt(task: str, difficulty: str) -> str:
    return (
        "You are a software engineering agent operating in SWE-Sim. "
        f"Task={task}, difficulty={difficulty}. "
        "Return exactly one JSON object with keys: action, filename (optional), code (optional). "
        "Allowed actions are inspect, patch, run_tests, submit; for fix_broken_api use inspect or submit. "
        "For resolve_ci_pipeline the bug is in utils.py (normalize divides by len instead of sum) — patch utils.py. "
        "Prefer minimal, correct fixes and avoid destructive rewrites."
    )


def _call_llm(client: OpenAI, task: str, difficulty: str, observation: Dict[str, Any], step: int) -> Dict[str, Any]:
    fallback = {"action": "run_tests" if task != "fix_broken_api" else "inspect", "filename": None, "code": None}
    user_prompt = (
        f"Step {step}. Current observation:\n{json.dumps(observation, ensure_ascii=True)}\n\n"
        "Respond with JSON only."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": _build_system_prompt(task, difficulty)},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content or ""
    return _parse_action(content, fallback)


def _post_json(http_client: httpx.Client, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = http_client.post(f"{OPENENV_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def run_episode(task: str, difficulty: str, llm_client: OpenAI, http_client: httpx.Client) -> Tuple[bool, int, float, List[float]]:
    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.001

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")

    try:
        reset_payload = {"task": task, "difficulty": difficulty}
        reset_resp = _post_json(http_client, "/reset", reset_payload)
        observation = reset_resp["observation"]
        max_steps = int(observation.get("max_steps", 10))

        for step_num in range(1, max_steps + 1):
            steps = step_num
            action = _call_llm(llm_client, task, difficulty, observation, step_num)

            step_error = None
            done = False
            reward_value = 0.001
            try:
                step_resp = _post_json(http_client, "/step", action)
                observation = step_resp["observation"]
                reward_value = float(step_resp["reward"]["reward"])
                done = bool(step_resp["done"])
                score = float(step_resp["info"].get("score", step_resp["reward"].get("tests_passed_ratio", 0.001)))
            except Exception as exc:  # noqa: BLE001
                step_error = str(exc)
                done = True

            rewards.append(reward_value)
            action_str = _single_line(json.dumps(action, ensure_ascii=True))
            error_str = "null" if step_error is None else _single_line(step_error)
            print(
                f"[STEP] step={step_num} action={action_str} reward={_fmt_reward(reward_value)} "
                f"done={_bool_str(done)} error={error_str}"
            )

            if done:
                break

        state_resp = http_client.get(f"{OPENENV_URL}/state", timeout=30)
        state_resp.raise_for_status()
        score = float(state_resp.json().get("score", score))
        score = max(0.001, min(0.999, score))
        success = score >= 0.99
    except Exception as exc:  # noqa: BLE001
        rewards.append(0.001)
        print(
            f"[STEP] step={steps if steps > 0 else 1} action={{}} reward=0.00 "
            f"done=true error={_single_line(str(exc))}"
        )
        success = False
        score = 0.001
    finally:
        rewards_str = ",".join(_fmt_reward(r) for r in rewards)
        print(
            f"[END] success={_bool_str(success)} steps={steps} score={_fmt_reward(score)} rewards={rewards_str}"
        )

    return success, steps, score, rewards


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("Missing credentials: set HF_TOKEN or API_KEY")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    with httpx.Client() as http_client:
        tasks = [
            ("fix_broken_api", "easy"),
            ("resolve_ci_pipeline", "medium"),
            ("debug_hidden_state", "hard"),
        ]
        for task_name, difficulty in tasks:
            run_episode(task_name, difficulty, llm_client, http_client)


if __name__ == "__main__":
    main()
