"""Rule-based fake chat client for demos and tests (no API key, no network).

The reply is a pure function of the conversation so far, so one client can
serve agents running in parallel. It is built from the framework's own
layers, so function calling, chat middleware and telemetry behave as with a
real provider.
"""

import asyncio
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
)
from agent_framework.observability import ChatTelemetryLayer

ToolCall = Tuple[str, Dict[str, Any]]
# policy(messages) -> {"thinking"?: str, "text"?: str, "tools"?: [(name, arguments)]}
Policy = Callable[[List[Message]], Dict[str, Any]]


class RuleBasedChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient):
    def __init__(self, policy: Policy, model: str = "demo-gpt-5", delay: Callable[[], float] = lambda: 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.policy = policy
        self.model = model
        self.delay = delay

    def _contents(self, messages: List[Message]) -> List[Content]:
        reply = self.policy(messages)
        contents: List[Content] = []
        if reply.get("thinking"):
            contents.append(Content.from_text_reasoning(text=reply["thinking"]))
        if reply.get("text"):
            contents.append(Content.from_text(reply["text"]))
        for name, arguments in reply.get("tools", []):
            contents.append(Content.from_function_call("call_" + uuid.uuid4().hex[:12], name, arguments=json.dumps(arguments)))
        return contents

    def _inner_get_response(self, *, messages: Any, stream: bool, options: Any, **kwargs: Any) -> Any:
        messages = list(messages)
        chars = sum(len(m.text or "") for m in messages)
        usage = {"input_token_count": 1400 + chars // 3, "output_token_count": 80}

        if stream:
            async def updates() -> Any:
                await asyncio.sleep(self.delay())
                yield ChatResponseUpdate(role="assistant", contents=self._contents(messages), model=self.model)
                yield ChatResponseUpdate(contents=[Content.from_usage(usage)])
            return self._build_response_stream(updates())

        async def response() -> ChatResponse:
            await asyncio.sleep(self.delay())
            return ChatResponse(messages=[Message("assistant", self._contents(messages))], model=self.model, usage_details=usage)
        return response()


def task_text(messages: Sequence[Message]) -> str:
    """Text of the latest user message."""
    for message in reversed(messages):
        if str(message.role) == "user" and message.text:
            return message.text
    return ""


def tool_results(messages: Sequence[Message], names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Tool results since the latest user message: name, result, error."""
    calls: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    for message in messages:
        if str(message.role) == "user" and message.text:
            results = []
        for content in message.contents or []:
            if content.type == "function_call":
                calls[content.call_id] = content.name
            elif content.type == "function_result":
                name = calls.get(content.call_id, "")
                if names is None or name in names:
                    results.append({"name": name, "result": content.result, "error": content.exception is not None})
    return results


def react(plan: Callable[[str], List[List[ToolCall]]], final: Callable[[str, List[Dict[str, Any]]], str],
          thinking: Optional[Callable[[str], str]] = None) -> Policy:
    """ReAct policy: run ``plan(task)`` turn by turn (a turn may hold several
    parallel tool calls), retry a failed call once, then answer with ``final``."""

    def policy(messages: List[Message]) -> Dict[str, Any]:
        task = task_text(messages)
        turns = plan(task)
        own = {n for turn in turns for n, _ in turn}
        results = tool_results(messages, list(own))
        errors: Dict[str, int] = {}
        done = 0
        for r in results:
            if r["error"]:
                errors[r["name"]] = errors.get(r["name"], 0) + 1
                if errors[r["name"]] < 2:
                    continue
            done += 1
        if results and results[-1]["error"] and errors[results[-1]["name"]] < 2:
            call = next(((n, a) for turn in turns for n, a in turn if n == results[-1]["name"]), None)
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
