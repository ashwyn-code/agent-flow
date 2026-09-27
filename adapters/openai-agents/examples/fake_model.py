"""Rule-based fake model for the OpenAI Agents SDK (no API key, no network).

The reply is a pure function of the conversation so far, so one model can
serve concurrent runs. It emits a generation span with its model name, output
and token usage, like the SDK's chat-completions model does.
"""

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tracing import generation_span
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import Summary

ToolCall = Tuple[str, Dict[str, Any]]
# policy(input items) -> {"thinking"?: str, "text"?: str, "tools"?: [(name, arguments)]}
Policy = Callable[[List[Any]], Dict[str, Any]]

TOOL_ERROR_PREFIX = "An error occurred while running the tool"


def _items(reply: Dict[str, Any]) -> List[Any]:
    out: List[Any] = []
    if reply.get("thinking"):
        out.append(ResponseReasoningItem(id=f"rs_{uuid.uuid4().hex[:10]}", type="reasoning",
                                         summary=[Summary(text=reply["thinking"], type="summary_text")]))
    for name, arguments in reply.get("tools", []):
        call_id = f"call_{uuid.uuid4().hex[:10]}"
        out.append(ResponseFunctionToolCall(type="function_call", id=call_id, call_id=call_id, name=name,
                                            arguments=json.dumps(arguments)))
    if reply.get("text") and not reply.get("tools"):
        out.append(ResponseOutputMessage(id=f"msg_{uuid.uuid4().hex[:10]}", type="message", role="assistant", status="completed",
                                         content=[ResponseOutputText(type="output_text", text=reply["text"], annotations=[])]))
    return out


class RuleBasedModel(Model):
    def __init__(self, policy: Policy, name: str = "demo-gpt-5-mini", delay: Callable[[], float] = lambda: 0.0) -> None:
        self.policy = policy
        self.name = name
        self.delay = delay

    async def _respond(self, input: Any, tracing: Any) -> Tuple[List[Any], Usage]:
        items = input if isinstance(input, list) else [{"role": "user", "content": input}]
        with generation_span(model=self.name, input=items if tracing.include_data() else None) as span:
            await asyncio.sleep(self.delay())
            out = _items(self.policy(items))
            chars = sum(len(json.dumps(i, default=str)) for i in items)
            usage = Usage(requests=1, input_tokens=900 + chars // 4, output_tokens=70, total_tokens=970 + chars // 4)
            if tracing.include_data():
                span.span_data.output = [o.model_dump() for o in out]
            span.span_data.usage = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
        return out, usage

    async def get_response(self, system_instructions: Any, input: Any, model_settings: Any, tools: Any, output_schema: Any,
                           handoffs: Any, tracing: Any, **kwargs: Any) -> ModelResponse:
        out, usage = await self._respond(input, tracing)
        return ModelResponse(output=out, usage=usage, response_id=f"resp_{uuid.uuid4().hex[:10]}")

    async def stream_response(self, system_instructions: Any, input: Any, model_settings: Any, tools: Any, output_schema: Any,
                              handoffs: Any, tracing: Any, **kwargs: Any) -> AsyncIterator[Any]:
        out, _ = await self._respond(input, tracing)
        response = Response(id=f"resp_{uuid.uuid4().hex[:10]}", created_at=0, model=self.name, object="response", output=out,
                            parallel_tool_calls=True, tool_choice="auto", tools=[])
        yield ResponseCompletedEvent(type="response.completed", response=response, sequence_number=0)


def task_text(items: Sequence[Any]) -> str:
    for item in items:
        if isinstance(item, dict) and item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
            return " ".join(p.get("text", "") for p in content or [] if isinstance(p, dict))
    return ""


def tool_results(items: Sequence[Any], names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Tool outputs so far: name, output, error."""
    calls: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            calls[item.get("call_id", "")] = item.get("name", "")
        elif item.get("type") == "function_call_output":
            name = calls.get(item.get("call_id", ""), "")
            output = str(item.get("output", ""))
            if names is None or name in names:
                results.append({"name": name, "output": output, "error": output.startswith(TOOL_ERROR_PREFIX)})
    return results


def react(plan: Callable[[str], List[List[ToolCall]]], final: Callable[[str, List[Dict[str, Any]]], str],
          thinking: Optional[Callable[[str], str]] = None) -> Policy:
    """Run ``plan(task)`` turn by turn (a turn may hold several parallel tool
    calls, or a ``transfer_to_*`` handoff), retry a failed call once, then
    answer with ``final``."""

    def policy(items: List[Any]) -> Dict[str, Any]:
        task = task_text(items)
        turns = plan(task)
        results = tool_results(items, [n for turn in turns for n, _ in turn])
        ok: Dict[str, int] = {}
        failed: Dict[str, int] = {}
        for r in results:
            bucket = failed if r["error"] else ok
            bucket[r["name"]] = bucket.get(r["name"], 0) + 1
        for index, turn in enumerate(turns):
            retry, fresh = [], []
            for name, arguments in turn:
                if ok.get(name, 0) > 0:
                    ok[name] -= 1          # done
                elif failed.get(name, 0) >= 2:
                    failed[name] -= 2      # failed twice: give up on it
                elif failed.get(name, 0) == 1:
                    failed[name] -= 1
                    retry.append((name, arguments))
                else:
                    fresh.append((name, arguments))
            if fresh:
                reply: Dict[str, Any] = {"tools": fresh + retry}
                if thinking is not None and index == 0 and not results:
                    reply["thinking"] = thinking(task)
                return reply
            if retry:
                return {"thinking": "That call failed; trying it again.", "tools": retry}
        return {"text": final(task, results)}

    return policy
