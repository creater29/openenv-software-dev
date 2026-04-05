"""
LLM-based semantic grader (optional).

When enabled, this grader sends the task description and agent's solution to
an LLM endpoint and asks for a structured quality score.  It is intentionally
lightweight — the LLM acts as a code reviewer, not as a test runner.

If the endpoint is unavailable or the request fails, the grader returns a
neutral score of 0.5 so it does not crash the episode.
"""
import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


class LLMGrader:
    """
    Optional semantic grader backed by any OpenAI-compatible chat endpoint.

    The prompt asks the model to return JSON with keys:
      score         : float 0.0 – 1.0
      justification : string
    """

    SYSTEM_PROMPT = (
        "You are an expert code reviewer grading a software engineering task. "
        "Return ONLY a JSON object with two keys: "
        "'score' (float 0.0 to 1.0) and 'justification' (one sentence). "
        "Do not include markdown fences or any other text."
    )

    def __init__(
        self,
        endpoint: str = "https://router.huggingface.co/v1/chat/completions",
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        api_key: Optional[str] = None,
        timeout: int = 15,
    ):
        self.endpoint = endpoint
        self.model    = model
        self.api_key  = api_key or "anonymous"
        self.timeout  = timeout

    def grade(
        self,
        task,
        vfs,
        executor,
    ) -> Dict[str, Any]:
        snapshot = vfs.snapshot()
        solution_code = snapshot.get("solution.py", "<no solution.py found>")

        user_prompt = (
            f"Task: {task.description}\n\n"
            f"Solution:\n```python\n{solution_code}\n```\n\n"
            "Rate the solution quality from 0.0 (completely wrong) to 1.0 (perfect)."
        )

        payload = json.dumps({
            "model": self.model,
            "max_tokens": 128,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        }).encode()

        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
            text = body["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            score = float(parsed.get("score", 0.5))
            justification = parsed.get("justification", "")
        except Exception as exc:
            score = 0.5
            justification = f"LLM grader unavailable: {exc}"

        return {
            "score":         round(score, 4),
            "justification": justification,
            "grader":        "llm",
        }
