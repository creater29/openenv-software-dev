"""
Baseline inference script for the OpenEnv Software Development environment.

This script is what judges run.  It follows the required STDOUT format exactly:
  [START] task=... env=... model=...
  [STEP]  step=N action=... reward=X.XX done=false error=null
  [END]   success=true steps=N score=X.XXX rewards=...

Usage:
  HF_TOKEN=hf_xxx python inference.py
"""
import os
import json
import asyncio
from openenv_software_dev.env import SoftwareDevEnv
from openenv_software_dev.observation import ObservationBuilder
from openenv_software_dev.actions import ActionType

# ── Try to import OpenAI client; fall back to rule-based agent ────────────────
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

OBS_BUILDER = ObservationBuilder()

# ── System prompt for the LLM agent ──────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert software engineer solving coding tasks in a virtual environment.
You can take the following actions (respond with ONLY valid JSON):
  {"type": 0, "target_file": "<name>", "text_input": ""}       -- READ_FILE
  {"type": 1, "target_file": "<name>", "text_input": "<code>"} -- WRITE_FILE
  {"type": 2, "target_file": "", "text_input": ""}             -- RUN_TESTS
  {"type": 3, "target_file": "", "text_input": ""}             -- SUBMIT
  {"type": 4, "target_file": "", "text_input": ""}             -- LIST_FILES

Strategy:
  1. Read the task description and existing files.
  2. Write a corrected or implemented solution.
  3. Run tests to verify.
  4. Submit when tests pass (or you are out of steps).
"""


def rule_based_action(obs: dict, step: int) -> dict:
    """Simple deterministic agent used as fallback when no API key is set."""
    files = obs.get("files", {})
    if step == 1:
        return {"type": ActionType.READ_FILE, "target_file": "solution.py", "text_input": ""}
    if step == 2:
        # Write a fixed add function (works for the default bugfix-001 task)
        return {
            "type": ActionType.WRITE_FILE,
            "target_file": "solution.py",
            "text_input": (
                "def add(a, b):\n    return a + b\n\n"
                "def multiply(a, b):\n    return a * b\n\n"
                "def is_palindrome(s):\n    return s == s[::-1]\n\n"
                "def factorial(n):\n"
                "    if n < 0:\n        raise ValueError('negative')\n"
                "    return 1 if n == 0 else n * factorial(n - 1)\n\n"
                "def fizzbuzz(n):\n"
                "    out = []\n"
                "    for i in range(1, n + 1):\n"
                "        if i % 15 == 0: out.append('FizzBuzz')\n"
                "        elif i % 3 == 0: out.append('Fizz')\n"
                "        elif i % 5 == 0: out.append('Buzz')\n"
                "        else: out.append(str(i))\n"
                "    return out\n\n"
                "def binary_search(arr, target):\n"
                "    lo, hi = 0, len(arr) - 1\n"
                "    while lo <= hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target: return mid\n"
                "        elif arr[mid] < target: lo = mid + 1\n"
                "        else: hi = mid - 1\n"
                "    return -1\n"
            ),
        }
    if step == 3:
        return {"type": ActionType.RUN_TESTS, "target_file": "", "text_input": ""}
    return {"type": ActionType.SUBMIT, "target_file": "", "text_input": ""}


def llm_action(client, model: str, obs: dict, history: list) -> dict:
    """Call the LLM to decide the next action."""
    obs_text = OBS_BUILDER.to_text(obs)
    history.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            max_tokens=512,
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": content})

        # Strip markdown fences if present
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
            content = content.rstrip("`").strip()

        action = json.loads(content)
        return action
    except Exception as exc:
        # On parse error, fall back to SUBMIT
        return {"type": ActionType.SUBMIT, "target_file": "", "text_input": "", "_error": str(exc)}


async def main():
    API_KEY    = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "anonymous")
    BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    MAX_STEPS  = int(os.getenv("MAX_STEPS", "30"))

    use_llm = HAS_OPENAI and API_KEY not in ("anonymous", "")
    client  = OpenAI(api_key=API_KEY, base_url=BASE_URL) if use_llm else None

    env = SoftwareDevEnv(max_steps=MAX_STEPS)
    obs, info = env.reset()

    print(
        f"[START] task={info['task_id']} env=SoftwareDev-v0 model={MODEL_NAME if use_llm else 'rule-based'}",
        flush=True,
    )

    total_reward = 0.0
    history = []
    step = 0
    done = trunc = False

    while not done and not trunc:
        step += 1

        # Choose action
        if use_llm:
            action_dict = llm_action(client, MODEL_NAME, obs, history)
        else:
            action_dict = rule_based_action(obs, step)

        action_str = json.dumps({k: v for k, v in action_dict.items() if k != "_error"})
        error_str  = action_dict.get("_error", "null")

        obs, reward, done, trunc, info = env.step(action_dict)
        total_reward += reward

        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward:.2f} done={str(done or trunc).lower()} error={error_str}",
            flush=True,
        )

    success = info.get("grading", {}).get("accepted", False)
    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={total_reward:.3f} rewards={total_reward:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
