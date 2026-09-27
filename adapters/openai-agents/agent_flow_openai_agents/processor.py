"""OpenAI Agents SDK trace processor that writes Agent Flow events.

Agent Flow's generic event source tails a JSONL file where each line is
``{"time": <seconds since start>, "type": <event type>, "payload": {...}}``
(see ``extension/src/protocol.ts`` and ``web/hooks/simulation/``).

Mapping from the Agents SDK to Agent Flow:

- A trace (one ``Runner.run``, or everything inside ``with trace(...)``) is
  the main agent. Its shape is a routing graph: every agent that runs is a
  node, handoffs are the hops between them (declared-but-unused handoffs
  are drawn dim), and parallel runs in one trace branch out from START.
- Each agent run is a subagent of the trace. An agent run from another
  agent's tool (``agent.as_tool()`` or your own) is a subagent of that agent.
- Function tool calls, handoffs and guardrails appear as tool calls
  (guardrails fail when their tripwire triggers); model calls give text,
  reasoning, model ids and token usage.

The SDK calls trace processors synchronously in the task that opens or closes
each span, and every span carries its parent, so events are written in order.
Note that ``set_tracing_disabled(True)`` also disables this processor; use
``install(..., exclusive=True)`` to drop the OpenAI exporter instead.
"""

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agents.tracing import TracingProcessor

from ._sink import EventSink

logger = logging.getLogger(__name__)

_MAX_CONTENT = 10_000
_MAX_ARGS = 200


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _output_parts(items: Any) -> Tuple[str, str]:
    """(assistant text, reasoning text) from Responses-style output items."""
    texts: List[str] = []
    thoughts: List[str] = []
    for item in items or []:
        kind = _get(item, "type")
        if kind == "message":
            for part in _get(item, "content") or []:
                if _get(part, "type") in ("output_text", "text") and _get(part, "text"):
                    texts.append(_get(part, "text"))
        elif kind == "reasoning":
            for part in (_get(item, "summary") or []) + (_get(item, "content") or []):
                if _get(part, "text"):
                    thoughts.append(_get(part, "text"))
        elif _get(item, "role") == "assistant" and isinstance(_get(item, "content"), str):
            texts.append(_get(item, "content"))  # chat-completions style
    return "\n".join(texts), "\n".join(thoughts)


def _first_user_text(items: Any) -> str:
    if isinstance(items, str):
        return items
    for item in items or []:
        if _get(item, "role") == "user":
            content = _get(item, "content")
            if isinstance(content, str):
                return content
            for part in content or []:
                if _get(part, "text"):
                    return _get(part, "text")
    return ""


class _TraceRun:
    """One trace's routing graph."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.completions: Dict[str, int] = {}
        self.consumed: Dict[Tuple[str, str], int] = {}
        self.node_step: Dict[str, int] = {}
        self.pending_handoffs: List[Tuple[str, str]] = []  # (from agent, to agent)
        self.last_completed: Optional[str] = None
        self.user_message_sent = False
        self.last_text: Dict[str, str] = {}


class AgentFlowProcessor(TracingProcessor):
    """Write Agent Flow JSONL events for OpenAI Agents SDK runs.

    Register it with :func:`install`, or yourself with
    ``agents.add_trace_processor`` / ``agents.set_trace_processors``.

    Args:
        path: JSONL file to append to. Defaults to ``$AGENT_FLOW_EVENT_LOG``,
            then ``agent-flow-events.jsonl`` in the current directory.
        truncate: Empty the file before the first event is written.
        url: Send events to an Agent Flow relay's ``/ingest`` endpoint instead of
            a file (defaults to ``$AGENT_FLOW_URL``); ``token`` authenticates it
            (``$AGENT_FLOW_TOKEN``) and ``session`` labels its tab.
        content: ``"metadata"`` replaces prompts, arguments, results and
            messages with their length (``$AGENT_FLOW_CONTENT``).
        redact: ``(event_type, payload) -> payload | None`` to scrub or drop events.
        sample_rate: Fraction of top-level runs to record (``$AGENT_FLOW_SAMPLE_RATE``).

    Events are delivered from a background thread; ``flush()`` waits for them
    and ``close()`` stops it (both also run at exit).
    """

    def __init__(self, path: Optional[str] = None, *, truncate: bool = False,
                 url: Optional[str] = None, token: Optional[str] = None, session: Optional[str] = None, content: Optional[str] = None, redact: Any = None, sample_rate: Optional[float] = None) -> None:
        self._sink = EventSink(path, truncate=truncate, url=url, token=token, session=session,
                               content=content, redact=redact, sample_rate=sample_rate)
        self.path = self._sink.path
        self._lock = threading.RLock()
        self._warned = False

        self._spans: Dict[str, Tuple[Optional[str], str]] = {}  # span id -> (parent id, span type)
        self._agents: Dict[str, str] = {}  # agent span id -> Agent Flow name
        self._holder: Set[str] = set()
        self._agent_parent: Dict[str, Optional[str]] = {}
        self._agent_node: Dict[str, Tuple[str, str]] = {}  # agent span id -> (trace id, node)
        self._traces: Dict[str, _TraceRun] = {}
        self._trace_names: Dict[str, str] = {}  # trace id -> Agent Flow name
        self._tools: Dict[str, Tuple[str, str]] = {}  # function span id -> (agent, tool)
        self._tool_started: Set[str] = set()
        self._models: Dict[str, str] = {}

    # ─── Output ──────────────────────────────────────────────────────────────

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until queued events have been written or sent."""
        self._sink.flush(timeout)

    def close(self) -> None:
        """Flush and stop the background writer."""
        self._sink.close()

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._sink.emit(event_type, payload)

    def _guard(self, fn: Any, *args: Any) -> None:
        try:
            with self._lock:
                fn(*args)
        except Exception:  # pragma: no cover - defensive
            if not self._warned:
                self._warned = True
                logger.exception("agent-flow: failed to record a trace event; further errors are suppressed")

    # ─── TracingProcessor ────────────────────────────────────────────────────

    def on_trace_start(self, trace: Any) -> None:
        self._guard(self._trace_started, trace)

    def on_trace_end(self, trace: Any) -> None:
        self._guard(self._trace_ended, trace)

    def on_span_start(self, span: Any) -> None:
        self._guard(self._span_started, span)

    def on_span_end(self, span: Any) -> None:
        self._guard(self._span_ended, span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    # ─── Names ───────────────────────────────────────────────────────────────

    def _claim(self, base: str) -> str:
        name, n = base, 2
        while name in self._holder:
            name, n = f"{base} #{n}", n + 1
        self._holder.add(name)
        return name

    def _release(self, name: str) -> None:
        self._holder.discard(name)

    # ─── Structure helpers ───────────────────────────────────────────────────

    def _ancestor(self, span_id: Optional[str], kinds: Sequence[str]) -> Optional[Tuple[str, str]]:
        """Nearest ancestor span of one of ``kinds``: (span id, type)."""
        hops = 0
        while span_id and hops < 200:
            parent_type = self._spans.get(span_id)
            if parent_type is None:
                return None
            if parent_type[1] in kinds:
                return span_id, parent_type[1]
            span_id = parent_type[0]
            hops += 1
        return None

    def _owner(self, span: Any) -> Optional[str]:
        """The agent a tool / model / guardrail span belongs to."""
        found = self._ancestor(span.parent_id, ("agent",))
        return self._agents.get(found[0]) if found else None

    # ─── Traces ──────────────────────────────────────────────────────────────

    def _trace_started(self, trace: Any) -> None:
        name = self._claim(getattr(trace, "name", None) or "Agent workflow")
        self._trace_names[trace.trace_id] = name
        self._traces[trace.trace_id] = _TraceRun(name)
        self._emit("agent_spawn", {"name": name, "isMain": True, "task": ""})
        self._emit("graph_structure", {"agent": name, "nodes": [
            {"id": "__start__", "label": "__start__", "kind": "start"},
            {"id": "__end__", "label": "__end__", "kind": "end"},
        ], "edges": []})

    def _trace_ended(self, trace: Any) -> None:
        run = self._traces.pop(trace.trace_id, None)
        name = self._trace_names.pop(trace.trace_id, None)
        if run is None or name is None:
            return
        if run.completions:
            sinks = sorted(n for n, count in run.completions.items()
                           if not any(src == n and used == count for (src, _), used in run.consumed.items()))
            if not sinks and run.last_completed:
                sinks = [run.last_completed]
            step = max(run.node_step.values(), default=0) + 1
            self._emit("node_start", {"agent": name, "node": "__end__", "step": step, "from": sinks})
        self._emit("agent_complete", {"name": name})
        self._release(name)

    # ─── Spans ───────────────────────────────────────────────────────────────

    def _span_started(self, span: Any) -> None:
        data = span.span_data
        kind = getattr(data, "type", "")
        self._spans[span.span_id] = (span.parent_id, kind)
        if kind == "agent":
            self._agent_started(span)
        elif kind == "function":
            self._function_started(span)
        elif kind == "generation":
            self._model_started(span, getattr(data, "model", None))

    def _span_ended(self, span: Any) -> None:
        kind = getattr(span.span_data, "type", "")
        handler = {
            "agent": self._agent_ended,
            "function": self._function_ended,
            "handoff": self._handoff_ended,
            "guardrail": self._guardrail_ended,
            "generation": self._generation_ended,
            "response": self._response_ended,
        }.get(kind)
        if handler is not None:
            handler(span)

    # Agents

    def _agent_started(self, span: Any) -> None:
        base = getattr(span.span_data, "name", None) or "agent"
        name = self._claim(base)
        self._agents[span.span_id] = name
        trace_name = self._trace_names.get(span.trace_id)
        tool = self._ancestor(span.parent_id, ("function",))
        if tool is not None and tool[0] in self._tools:
            # Run from another agent's tool: a subagent of that agent
            parent = self._tools[tool[0]][0]
            task = self._tools[tool[0]][1]
            self._agent_parent[name] = parent
            self._emit("subagent_dispatch", {"parent": parent, "child": name, "task": f"via {task}"})
            self._emit("agent_spawn", {"name": name, "parent": parent, "task": f"via {task}"})
            return
        if trace_name is None:
            return
        self._agent_parent[name] = trace_name
        self._emit("subagent_dispatch", {"parent": trace_name, "child": name, "task": base})
        self._emit("agent_spawn", {"name": name, "parent": trace_name, "task": base})
        run = self._traces[span.trace_id]
        # Reached through a handoff in this trace, else from START
        sources = []
        for i, (src, dst) in enumerate(run.pending_handoffs):
            if dst == base:
                sources.append(src)
                del run.pending_handoffs[i]
                break
        sources = sources or ["__start__"]
        for src in sources:
            if src != "__start__":
                run.consumed[(src, base)] = run.completions.get(src, 0)
        step = 1 + max((run.node_step.get(s, 0) for s in sources), default=0)
        run.node_step[base] = step
        self._agent_node[span.span_id] = (span.trace_id, base)
        self._emit("node_start", {"agent": trace_name, "node": base, "step": step, "from": sources})

    def _agent_ended(self, span: Any) -> None:
        name = self._agents.pop(span.span_id, None)
        if name is None:
            return
        data = span.span_data
        error = getattr(span, "error", None)
        node = self._agent_node.pop(span.span_id, None)
        if node is not None:
            trace_id, base = node
            run = self._traces.get(trace_id)
            trace_name = self._trace_names.get(trace_id)
            if run is not None and trace_name is not None:
                # Declare this agent's handoffs as (model-chosen, so conditional) routes
                handoffs = [h for h in (getattr(data, "handoffs", None) or []) if isinstance(h, str)]
                self._emit("graph_structure", {
                    "agent": trace_name,
                    "nodes": [{"id": base, "label": base, "kind": "node"}] + [{"id": h, "label": h, "kind": "node"} for h in handoffs],
                    "edges": [{"source": base, "target": h, "conditional": True} for h in handoffs],
                })
                run.completions[base] = run.completions.get(base, 0) + 1
                run.last_completed = base
                payload: Dict[str, Any] = {"agent": trace_name, "node": base, "step": run.node_step.get(base, 0)}
                if error:
                    payload["error"] = _truncate(_to_text(error.get("message") if isinstance(error, dict) else error), _MAX_CONTENT)
                self._emit("node_end", payload)
        summary = ""
        run = self._traces.get(span.trace_id)
        if run is not None:
            summary = run.last_text.pop(name, "")
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            self._emit("message", {"agent": name, "content": f"Error: {message}"})
            summary = summary or f"Error: {message}"
        parent = self._agent_parent.pop(name, None)
        if parent is not None:
            self._emit("subagent_return", {"parent": parent, "child": name, "summary": _truncate(summary, _MAX_CONTENT)})
        self._emit("agent_complete", {"name": name})
        self._release(name)

    # Tools, handoffs, guardrails

    def _function_started(self, span: Any) -> None:
        agent = self._owner(span)
        if agent is None:
            return
        self._tools[span.span_id] = (agent, getattr(span.span_data, "name", "tool"))
        # The SDK fills the tool input just after opening the span; report the
        # call once the current step yields so the arguments are known.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.call_soon(lambda: self._guard(self._tool_call_start, span))
        else:
            self._tool_call_start(span)

    def _tool_call_start(self, span: Any) -> None:
        if span.span_id in self._tool_started or span.span_id not in self._tools:
            return
        self._tool_started.add(span.span_id)
        agent, tool = self._tools[span.span_id]
        raw = getattr(span.span_data, "input", None)
        payload: Dict[str, Any] = {"agent": agent, "tool": tool, "args": _truncate(_to_text(raw), _MAX_ARGS)}
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            payload["inputData"] = parsed
        self._emit("tool_call_start", payload)

    def _function_ended(self, span: Any) -> None:
        if span.span_id not in self._tools:
            return
        self._tool_call_start(span)  # a tool that finished before the loop got a turn
        agent, tool = self._tools.pop(span.span_id)
        self._tool_started.discard(span.span_id)
        output = _to_text(getattr(span.span_data, "output", None))
        payload: Dict[str, Any] = {"agent": agent, "tool": tool, "result": _truncate(output, _MAX_CONTENT)}
        error = getattr(span, "error", None)
        if error:
            detail = (error.get("data") or {}).get("error") if isinstance(error, dict) else None
            payload["isError"] = True
            payload["errorMessage"] = _truncate(_to_text(detail or (error.get("message") if isinstance(error, dict) else error)), _MAX_CONTENT)
        self._emit("tool_call_end", payload)

    def _handoff_ended(self, span: Any) -> None:
        data = span.span_data
        source, target = getattr(data, "from_agent", None), getattr(data, "to_agent", None)
        agent = self._owner(span)
        if agent is not None:
            tool = f"handoff → {target}" if target else "handoff"
            self._emit("tool_call_start", {"agent": agent, "tool": tool, "args": _to_text({"to": target})})
            self._emit("tool_call_end", {"agent": agent, "tool": tool, "result": f"transferred to {target}"})
        run = self._traces.get(span.trace_id)
        if run is not None and source and target:
            run.pending_handoffs.append((source, target))

    def _guardrail_ended(self, span: Any) -> None:
        agent = self._owner(span)
        if agent is None:
            return
        data = span.span_data
        tool = f"guardrail: {getattr(data, 'name', '')}"
        triggered = bool(getattr(data, "triggered", False))
        self._emit("tool_call_start", {"agent": agent, "tool": tool, "args": ""})
        payload: Dict[str, Any] = {"agent": agent, "tool": tool, "result": "tripwire triggered" if triggered else "passed"}
        if triggered:
            payload["isError"] = True
            payload["errorMessage"] = "tripwire triggered"
        self._emit("tool_call_end", payload)

    # Model calls

    def _model_started(self, span: Any, model: Any) -> None:
        agent = self._owner(span)
        if agent is not None and isinstance(model, str) and model and self._models.get(agent) != model:
            self._models[agent] = model
            self._emit("model_detected", {"agent": agent, "model": model})

    def _model_output(self, span: Any, output: Any, usage: Any, input_items: Any) -> None:
        agent = self._owner(span)
        if agent is None:
            return
        run = self._traces.get(span.trace_id)
        trace_name = self._trace_names.get(span.trace_id)
        if run is not None and trace_name is not None and not run.user_message_sent:
            prompt = _first_user_text(input_items)
            if prompt:
                run.user_message_sent = True
                self._emit("message", {"agent": trace_name, "role": "user", "content": _truncate(prompt, _MAX_CONTENT)})
        text, thought = _output_parts(output)
        if thought:
            self._emit("message", {"agent": agent, "role": "thinking", "content": _truncate(thought, _MAX_CONTENT)})
        if text.strip():
            self._emit("message", {"agent": agent, "role": "assistant", "content": _truncate(text, _MAX_CONTENT)})
            if run is not None:
                run.last_text[agent] = text
        tokens = 0
        if isinstance(usage, dict):
            tokens = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        elif usage is not None:
            tokens = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
        if tokens:
            self._emit("context_update", {"agent": agent, "tokens": tokens})

    def _generation_ended(self, span: Any) -> None:
        data = span.span_data
        self._model_output(span, getattr(data, "output", None), getattr(data, "usage", None), getattr(data, "input", None))

    def _response_ended(self, span: Any) -> None:
        data = span.span_data
        response = getattr(data, "response", None)
        self._model_started(span, getattr(response, "model", None))
        self._model_output(span, getattr(response, "output", None), getattr(data, "usage", None) or getattr(response, "usage", None),
                           getattr(data, "input", None))


def install(path: Optional[str] = None, *, truncate: bool = False, exclusive: bool = False) -> AgentFlowProcessor:
    """Create a processor and register it with the Agents SDK.

    By default it is added next to the SDK's own processors, so traces still
    go to the OpenAI dashboard if you use it. With ``exclusive=True`` it
    replaces them, which also stops the SDK from uploading traces (useful
    offline, in tests, or without an OpenAI key).
    """
    from agents import add_trace_processor, set_trace_processors

    processor = AgentFlowProcessor(path, truncate=truncate)
    if exclusive:
        set_trace_processors([processor])
    else:
        add_trace_processor(processor)
    return processor


__all__: Sequence[str] = ["AgentFlowProcessor", "install"]
