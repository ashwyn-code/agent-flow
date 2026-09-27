"""Rule-based fake ADK model for demos and tests (no API key, no network).

The reply is a pure function of the request so far, so one model can serve
concurrent runs (ParallelAgent children, parallel AgentTool calls).
"""

import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

ToolCall = Tuple[str, Dict[str, Any]]
# policy(contents) -> {"thinking"?: str, "text"?: str, "tools"?: [(name, args)]}
Policy = Callable[[List[types.Content]], Dict[str, Any]]


class RuleBasedLlm(BaseLlm):
    policy: Any = None
    delay: Any = None  # Callable[[], float]

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        if self.delay is not None:
            await asyncio.sleep(self.delay())
        contents = list(getattr(llm_request, "contents", None) or [])
        reply = self.policy(contents)
        parts: List[types.Part] = []
        if reply.get("thinking"):
            parts.append(types.Part(text=reply["thinking"], thought=True))
        if reply.get("text"):
            parts.append(types.Part(text=reply["text"]))
        for name, args in reply.get("tools", []):
            parts.append(types.Part(function_call=types.FunctionCall(name=name, args=args)))
        chars = sum(len(str(c)) for c in contents)
        yield LlmResponse(
            content=types.Content(role="model", parts=parts), partial=False, turn_complete=True, model_version=self.model,
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1100 + chars // 8, candidates_token_count=60, total_token_count=1160 + chars // 8),
        )


def fake_llm(policy: Policy, model: str = "demo-gemini-2.5-flash", delay: Optional[Callable[[], float]] = None) -> RuleBasedLlm:
    return RuleBasedLlm(model=model, policy=policy, delay=delay)


def task_text(contents: Sequence[types.Content]) -> str:
    """The latest user text (for an agent tool, the request it was given)."""
    for content in reversed(contents):
        if content.role == "user":
            for part in content.parts or []:
                if part.text and not part.function_response:
                    return part.text
    return ""


def tool_results(contents: Sequence[types.Content], names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Function responses so far: name, response, error."""
    results: List[Dict[str, Any]] = []
    for content in contents:
        for part in content.parts or []:
            response = part.function_response
            if response is not None and (names is None or response.name in names):
                body = response.response or {}
                results.append({"name": response.name, "response": body,
                                "error": isinstance(body, dict) and body.get("status") == "error"})
    return results


def react(plan: Callable[[str], List[List[ToolCall]]], final: Callable[[str, List[Dict[str, Any]]], str],
          thinking: Optional[Callable[[str], str]] = None) -> Policy:
    """Run ``plan(task)`` turn by turn (a turn may hold several parallel tool
    calls), retry a call that reported an error once, then answer."""

    def policy(contents: List[types.Content]) -> Dict[str, Any]:
        task = task_text(contents)
        turns = plan(task)
        # A loop re-runs the agent over the same history: only count tool
        # results since its last final answer.
        start = 0
        for i, content in enumerate(contents):
            parts = content.parts or []
            if content.role == "model" and any(p.text and not p.thought for p in parts) and not any(p.function_call for p in parts):
                start = i + 1
        results = tool_results(contents[start:], [n for turn in turns for n, _ in turn])
        ok: Dict[str, int] = {}
        failed: Dict[str, int] = {}
        for r in results:
            bucket = failed if r["error"] else ok
            bucket[r["name"]] = bucket.get(r["name"], 0) + 1
        for index, turn in enumerate(turns):
            retry, fresh = [], []
            for name, args in turn:
                if ok.get(name, 0) > 0:
                    ok[name] -= 1
                elif failed.get(name, 0) >= 2:
                    failed[name] -= 2
                elif failed.get(name, 0) == 1:
                    failed[name] -= 1
                    retry.append((name, args))
                else:
                    fresh.append((name, args))
            if fresh:
                reply: Dict[str, Any] = {"tools": fresh + retry}
                if thinking is not None and index == 0 and not results:
                    reply["thinking"] = thinking(task)
                return reply
            if retry:
                return {"thinking": "That call failed; trying again.", "tools": retry}
        return {"text": final(task, results)}

    return policy
