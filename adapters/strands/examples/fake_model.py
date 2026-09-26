"""Rule-based fake Strands model for demos and tests (no API key).

The reply is a pure function of the conversation so far, so one model
instance is safe to share across agents running in parallel.
"""

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

from strands.models.model import Model

ToolCall = Tuple[str, Dict[str, Any]]


# policy(messages, tool_names) -> {"thinking"?: str, "text"?: str, "tools"?: [(name, input)]}
Policy = Callable[[List[Dict[str, Any]], Sequence[str]], Dict[str, Any]]


class RuleBasedModel(Model):
    def __init__(self, policy: Policy, model_id: str = "demo-claude-sonnet-4-5", delay: Callable[[], float] = lambda: 0.0):
        self.policy = policy
        self.delay = delay
        self.config: Dict[str, Any] = {"model_id": model_id}

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    async def structured_output(self, output_model: Any, prompt: Any, system_prompt: Optional[str] = None, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError("structured output is not supported by the fake model")
        yield  # pragma: no cover

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Optional[str] = None, **kwargs: Any) -> AsyncIterator[Any]:
        await asyncio.sleep(self.delay())
        names = [spec["name"] for spec in (tool_specs or [])]
        reply = self.policy(messages, names)
        chars = sum(len(json.dumps(m.get("content", ""), default=str)) for m in messages)

        yield {"messageStart": {"role": "assistant"}}
        if reply.get("thinking"):
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": reply["thinking"]}}}}
            yield {"contentBlockStop": {}}
        if reply.get("text"):
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": reply["text"]}}}
            yield {"contentBlockStop": {}}
        for name, tool_input in reply.get("tools", []):
            yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": "tooluse_" + uuid.uuid4().hex[:16]}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_input)}}}}
            yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use" if reply.get("tools") else "end_turn"}}
        input_tokens = 1500 + chars // 3
        yield {"metadata": {"usage": {"inputTokens": input_tokens, "outputTokens": 90, "totalTokens": input_tokens + 90},
                            "metrics": {"latencyMs": 40}}}


def task_text(messages: List[Dict[str, Any]]) -> str:
    """Text of the latest user message that isn't a tool result."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        texts = [b["text"] for b in message.get("content", []) if "text" in b]
        if texts:
            return "\n".join(texts)
    return ""


def tool_results(messages: List[Dict[str, Any]], names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Tool results since the latest user text, with the tool name attached."""
    uses: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    for message in messages:
        for block in message.get("content", []):
            if message.get("role") == "user" and "text" in block:
                results = []
            if "toolUse" in block:
                uses[block["toolUse"]["toolUseId"]] = block["toolUse"]["name"]
            if "toolResult" in block:
                result = dict(block["toolResult"])
                result["name"] = uses.get(result.get("toolUseId", ""), "")
                if names is None or result["name"] in names:
                    results.append(result)
    return results


def react(plan: Callable[[str], List[List[ToolCall]]], final: Callable[[str, List[Dict[str, Any]]], str],
          thinking: Optional[Callable[[str], str]] = None) -> Policy:
    """ReAct policy: run ``plan(task)`` turn by turn (a turn may hold several
    parallel tool calls), re-issue any call whose result was an error, then
    answer with ``final``."""

    def policy(messages: List[Dict[str, Any]], names: Sequence[str]) -> Dict[str, Any]:
        task = task_text(messages)
        turns = plan(task)
        own = {n for turn in turns for n, _ in turn}
        results = tool_results(messages, list(own))
        # Retry a call that just failed, once; after a second failure give up on it
        errors: Dict[str, int] = {}
        done = 0
        for r in results:
            if r.get("status") == "error":
                errors[r["name"]] = errors.get(r["name"], 0) + 1
                if errors[r["name"]] < 2:
                    continue
            done += 1
        if results and results[-1].get("status") == "error" and errors[results[-1]["name"]] < 2:
            last_failed = results[-1]["name"]
            call = next(((n, a) for turn in turns for n, a in turn if n == last_failed), None)
            if call is not None:
                return {"text": "That failed, retrying.", "tools": [call]}
        seen = 0
        for turn in turns:
            if done < seen + len(turn):
                reply: Dict[str, Any] = {"tools": turn}
                if thinking is not None and seen == 0:
                    reply["thinking"] = thinking(task)
                return reply
            seen += len(turn)
        return {"text": final(task, results)}

    return policy
