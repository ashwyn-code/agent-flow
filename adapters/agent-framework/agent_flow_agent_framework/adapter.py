"""Microsoft Agent Framework adapter that writes Agent Flow events.

Agent Flow's generic event source tails a JSONL file where each line is
``{"time": <seconds since start>, "type": <event type>, "payload": {...}}``
(see ``extension/src/protocol.ts`` and ``web/hooks/simulation/``).

Mapping from Agent Framework to Agent Flow:

- Every ``Agent`` run is an Agent Flow agent, observed through agent, chat
  and function middleware. The first one with no caller is the main agent.
- An agent run from inside another agent's tool (``agent.as_tool()`` or your
  own tool function) is a subagent of that agent.
- A ``Workflow`` is an agent whose shape is a node graph: its executors and
  edge groups are sent as ``graph_structure`` and each executor execution as
  ``node_start`` / ``node_end`` with the exact executors whose messages it
  received, so fan-out, fan-in, switch-case routing, loops and nested
  workflows are drawn as they ran. Agents running in ``AgentExecutor``
  nodes are its subagents, named after their executor.
- Tool calls, model text, reasoning, model ids and token usage map to the
  existing event types.

Workflows have no public observer API, so ``instrument(workflow)`` wraps
that workflow's ``run`` and its executors' ``execute`` methods (instance
attributes only; classes are untouched). Callers are linked with context
variables, which the framework copies into the tasks and threads it runs
tools and executors in.
"""

import contextlib
import contextvars
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from agent_framework import AgentMiddleware, ChatMiddleware, FunctionMiddleware

from ._sink import EventSink

logger = logging.getLogger(__name__)

_MAX_CONTENT = 10_000
_MAX_ARGS = 200


@dataclass(frozen=True)
class _Frame:
    """What started the code running now: a tool call or a workflow node."""

    kind: str  # "tool" | "node"
    parent: str  # Agent Flow name of the owning agent / workflow
    label: str  # tool name or executor id


_FRAME: "contextvars.ContextVar[Optional[_Frame]]" = contextvars.ContextVar("agent_flow_frame", default=None)
# Agent Flow name of the agent whose model / tool calls are running now
_AGENT: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("agent_flow_agent", default=None)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str) and text:
        return text
    if isinstance(value, (list, tuple)):
        return "\n".join(t for t in (_to_text(v) for v in value) if t)
    result = getattr(value, "result", None)
    if result is not None and result is not value:
        return _to_text(result)
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _last_user_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    for message in reversed(list(messages or [])):
        role = getattr(message, "role", None)
        if str(role) == "user" or getattr(role, "value", None) == "user":
            text = getattr(message, "text", "")
            if text:
                return text
    return ""


def _message_text(message: Any) -> str:
    """Text of a workflow input: str, Message, list of messages, request objects..."""
    if isinstance(message, str):
        return message
    for attr in ("text", "messages", "agent_response"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list):
            return _last_user_text(value) or _to_text(value)
        if value is not None and value is not message:
            return _message_text(value)
    if isinstance(message, list):
        return "\n".join(t for t in (_message_text(m) for m in message) if t)
    return _to_text(message)


def _arguments(arguments: Any) -> Any:
    if hasattr(arguments, "model_dump"):
        return arguments.model_dump()
    try:
        return dict(arguments)
    except (TypeError, ValueError):
        return arguments


class _RunState:
    """Per workflow run: which executor outputs have not flowed anywhere yet."""

    def __init__(self) -> None:
        self.completions: Dict[str, int] = {}
        self.consumed: Dict[Tuple[str, str], int] = {}
        self.node_step: Dict[str, int] = {}
        self.last_completed: Optional[str] = None


class AgentFlow:
    """Write Agent Flow JSONL events for Microsoft Agent Framework agents and workflows.

    Usage::

        flow = AgentFlow("agent-flow.jsonl", truncate=True)
        agent = Agent(client, name="helper", tools=[...], middleware=flow.middleware)
        # or attach to existing agents / workflows (recursing into their executors):
        flow.instrument(workflow)

    One instance is one Agent Flow session. Agents that only exist inside
    your own tool functions need ``middleware=flow.middleware`` (or
    ``flow.instrument(agent)``) too; ``agent.as_tool()`` agents passed to an
    instrumented agent are picked up automatically.

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

        # Names are held only while a run is in flight: a later run reuses the
        # name, concurrent runs (e.g. one agent called as a tool 3x at once)
        # get "name #2", ... Agent runs are keyed by their middleware context,
        # workflow runs by the workflow (a workflow can't run concurrently).
        self._names: Dict[int, str] = {}  # id(run key) -> name
        self._holder: Dict[str, int] = {}
        self._parent: Dict[str, Optional[str]] = {}
        self._user_message_sent = False
        self._structure_sent: Set[str] = set()
        self._models: Dict[str, str] = {}
        self._runs: Dict[str, _RunState] = {}
        self._instrumented: Set[int] = set()
        self.middleware: List[Any] = [_AgentMiddleware(self), _FunctionMiddleware(self), _ChatMiddleware(self)]

    # ─── Attaching ───────────────────────────────────────────────────────────

    def instrument(self, target: Any) -> Any:
        """Attach to an Agent, AgentExecutor, WorkflowExecutor or Workflow and
        everything nested in it (agents used as tools included). Returns ``target``."""
        if id(target) in self._instrumented:
            return target
        self._instrumented.add(id(target))
        if hasattr(target, "executors") and hasattr(target, "edge_groups") and hasattr(target, "run"):
            self._instrument_workflow(target)
            return target
        inner_workflow = getattr(target, "workflow", None)
        if inner_workflow is not None and hasattr(inner_workflow, "executors"):
            self.instrument(inner_workflow)
        agent = getattr(target, "agent", None) or getattr(target, "_agent", None)
        if agent is not None and agent is not target:
            self.instrument(agent)
        if hasattr(target, "middleware") and hasattr(target, "run") and not hasattr(target, "executors"):
            existing = list(target.middleware or [])
            if not any(m in existing for m in self.middleware):
                target.middleware = existing + self.middleware
                if hasattr(target, "_cached_agent_middleware_pipeline"):
                    target._cached_agent_middleware_pipeline = None
            options = getattr(target, "default_options", None) or {}
            for agent_tool in options.get("tools") or []:
                owner = _tool_agent(agent_tool)
                if owner is not None:
                    self.instrument(owner)
        return target

    def _guard(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run bookkeeping; never let the visualizer break the run."""
        try:
            with self._lock:
                return fn(*args)
        except Exception:  # pragma: no cover - defensive
            if not self._warned:
                self._warned = True
                logger.exception("agent-flow: event bookkeeping failed; further errors are suppressed")
            return None

    # ─── Output ──────────────────────────────────────────────────────────────

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until queued events have been written or sent."""
        self._sink.flush(timeout)

    def close(self) -> None:
        """Flush and stop the background writer."""
        self._sink.close()

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._sink.emit(event_type, payload)

    # ─── Agents and workflows starting / finishing ───────────────────────────

    def _name_for(self, key: int, obj: Any, preferred: Optional[str]) -> str:
        if key in self._names:
            return self._names[key]
        base = preferred or getattr(obj, "name", None) or getattr(obj, "id", None) or type(obj).__name__
        name, n = str(base), 2
        while name in self._holder:
            name, n = f"{base} #{n}", n + 1
        self._holder[name] = key
        self._names[key] = name
        return name

    def _begin(self, obj: Any, task: str, run_key: Any = None) -> str:
        """Announce a starting agent / workflow run; returns its name."""
        key = id(run_key if run_key is not None else obj)
        if key in self._names:
            return self._names[key]
        frame = _FRAME.get()
        preferred = frame.label if frame is not None and frame.kind == "node" else None
        name = self._name_for(key, obj, preferred)
        parent = frame.parent if frame is not None and frame.parent != name else None
        task = _truncate(task, _MAX_CONTENT)
        self._parent[name] = parent
        if parent is not None:
            self._emit("subagent_dispatch", {"parent": parent, "child": name, "task": task})
            self._emit("agent_spawn", {"name": name, "parent": parent, "task": task})
        else:
            self._emit("agent_spawn", {"name": name, "isMain": True, "task": task})
            if task and not self._user_message_sent:
                self._user_message_sent = True
                self._emit("message", {"agent": name, "role": "user", "content": task})
        return name

    def _finish(self, run_key: Any, summary: str, error: Optional[BaseException] = None) -> None:
        key = id(run_key)
        name = self._names.get(key)
        if name is None:
            return
        if error is not None:
            self._emit("message", {"agent": name, "content": f"Error: {error!r}"})
            summary = summary or f"Error: {error!r}"
        parent = self._parent.get(name)
        if parent is not None:
            self._emit("subagent_return", {"parent": parent, "child": name, "summary": _truncate(summary, _MAX_CONTENT)})
        self._emit("agent_complete", {"name": name})
        self._names.pop(key, None)
        if self._holder.get(name) == key:
            del self._holder[name]

    # ─── Workflows ───────────────────────────────────────────────────────────

    def _instrument_workflow(self, workflow: Any) -> None:
        original_run = workflow.run
        flow = self

        def run(*args: Any, **kwargs: Any) -> Any:
            message = args[0] if args else kwargs.get("message", kwargs.get("messages"))
            flow._guard(flow._workflow_started, workflow, message)
            try:
                result = original_run(*args, **kwargs)
            except BaseException as exc:
                flow._guard(flow._workflow_finished, workflow, exc)
                raise
            if kwargs.get("stream"):
                return result.with_cleanup_hook(lambda: flow._guard(flow._workflow_finished, workflow, None))

            async def awaited() -> Any:
                try:
                    value = await result
                except BaseException as exc:
                    flow._guard(flow._workflow_finished, workflow, exc)
                    raise
                flow._guard(flow._workflow_finished, workflow, None)
                return value

            return awaited()

        workflow.run = run
        for executor_id, executor in workflow.executors.items():
            self._wrap_executor(workflow, executor_id, executor)
            self.instrument(executor)

    def _workflow_started(self, workflow: Any, message: Any) -> None:
        name = self._begin(workflow, _message_text(message))
        self._runs[name] = _RunState()
        if name not in self._structure_sent:
            self._structure_sent.add(name)
            self._emit("graph_structure", {"agent": name, **_structure(workflow)})

    def _workflow_finished(self, workflow: Any, error: Optional[BaseException]) -> None:
        name = self._names.get(id(workflow))
        if name is None:
            return
        run = self._runs.pop(name, None)
        if run is not None and run.completions:
            sinks = sorted(n for n, count in run.completions.items()
                           if not any(src == n and used == count for (src, _), used in run.consumed.items()))
            if not sinks and run.last_completed:
                sinks = [run.last_completed]
            step = max(run.node_step.values(), default=0) + 1
            self._emit("node_start", {"agent": name, "node": "__end__", "step": step, "from": sinks})
        self._finish(workflow, "", error)

    def _wrap_executor(self, workflow: Any, executor_id: str, executor: Any) -> None:
        original = executor.execute
        flow = self

        async def execute(message: Any, source_executor_ids: Any = None, *args: Any, **kwargs: Any) -> Any:
            token = flow._guard(flow._node_started, workflow, executor_id, list(source_executor_ids or []))
            error: Optional[BaseException] = None
            try:
                return await original(message, source_executor_ids, *args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                if token is not None:
                    try:
                        _FRAME.reset(token)
                    except ValueError:
                        pass
                flow._guard(flow._node_finished, workflow, executor_id, error)

        executor.execute = execute

    def _node_started(self, workflow: Any, executor_id: str, sources: List[str]) -> Optional[contextvars.Token]:
        name = self._names.get(id(workflow))
        if name is None:
            return None
        run = self._runs.setdefault(name, _RunState())
        hops = sorted({"__start__" if s.startswith("internal:") else s for s in sources}) or ["__start__"]
        for src in hops:
            if src != "__start__":
                run.consumed[(src, executor_id)] = run.completions.get(src, 0)
        step = 1 + max((run.node_step.get(s, 0) for s in hops), default=0)
        run.node_step[executor_id] = step
        self._emit("node_start", {"agent": name, "node": executor_id, "step": step, "from": hops})
        return _FRAME.set(_Frame("node", name, executor_id))

    def _node_finished(self, workflow: Any, executor_id: str, error: Optional[BaseException]) -> None:
        name = self._names.get(id(workflow))
        if name is None:
            return
        run = self._runs.setdefault(name, _RunState())
        run.completions[executor_id] = run.completions.get(executor_id, 0) + 1
        run.last_completed = executor_id
        payload: Dict[str, Any] = {"agent": name, "node": executor_id, "step": run.node_step.get(executor_id, 0)}
        if error is not None:
            payload["error"] = _truncate(str(error) or type(error).__name__, _MAX_CONTENT)
        self._emit("node_end", payload)

    # ─── Model and tool calls ────────────────────────────────────────────────

    def _current_agent(self) -> Optional[str]:
        return _AGENT.get()

    def _model_call(self, client: Any, options: Any) -> Optional[str]:
        agent = self._current_agent()
        if agent is None:
            return None
        model = (options or {}).get("model") if isinstance(options, dict) else None
        model = model or getattr(client, "model", None) or getattr(client, "model_id", None)
        if isinstance(model, str) and model and self._models.get(agent) != model:
            self._models[agent] = model
            self._emit("model_detected", {"agent": agent, "model": model})
        return agent

    def _model_response(self, agent: str, response: Any) -> None:
        for message in getattr(response, "messages", None) or []:
            for content in getattr(message, "contents", None) or []:
                kind = getattr(content, "type", None)
                text = getattr(content, "text", None)
                if kind == "text_reasoning" and text:
                    self._emit("message", {"agent": agent, "role": "thinking", "content": _truncate(text, _MAX_CONTENT)})
                elif kind == "text" and text and text.strip():
                    self._emit("message", {"agent": agent, "role": "assistant", "content": _truncate(text, _MAX_CONTENT)})
        usage = getattr(response, "usage_details", None) or {}
        tokens = (usage.get("input_token_count") or 0) + (usage.get("output_token_count") or 0)
        if tokens:
            self._emit("context_update", {"agent": agent, "tokens": tokens})

    def _tool_started(self, name: str, arguments: Any) -> Optional[Tuple[str, contextvars.Token]]:
        agent = self._current_agent()
        if agent is None:
            return None
        data = _arguments(arguments)
        payload: Dict[str, Any] = {"agent": agent, "tool": name, "args": _truncate(_to_text(data), _MAX_ARGS)}
        if isinstance(data, dict):
            payload["inputData"] = data
        self._emit("tool_call_start", payload)
        return agent, _FRAME.set(_Frame("tool", agent, name))

    def _tool_finished(self, agent: str, name: str, result: Any, error: Optional[BaseException]) -> None:
        payload: Dict[str, Any] = {"agent": agent, "tool": name, "result": _truncate(_to_text(result), _MAX_CONTENT)}
        if error is not None:
            payload["isError"] = True
            payload["errorMessage"] = _truncate(str(error) or type(error).__name__, _MAX_CONTENT)
            payload["result"] = payload["errorMessage"]
        self._emit("tool_call_end", payload)


@contextlib.contextmanager
def _agent_scope(name: str) -> Iterator[None]:
    token = _AGENT.set(name)
    try:
        yield
    finally:
        _AGENT.reset(token)


def _tool_agent(agent_tool: Any) -> Optional[Any]:
    """The agent behind an ``agent.as_tool()`` FunctionTool, if any."""
    func = getattr(agent_tool, "func", None) or getattr(agent_tool, "_func", None)
    for cell in getattr(func, "__closure__", None) or ():
        value = cell.cell_contents
        if hasattr(value, "run") and hasattr(value, "middleware") and not hasattr(value, "executors"):
            return value
    return None


# ─── Middleware ──────────────────────────────────────────────────────────────

class _AgentMiddleware(AgentMiddleware):
    def __init__(self, flow: AgentFlow) -> None:
        self.flow = flow

    async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
        flow, agent = self.flow, context.agent
        name = flow._guard(flow._begin, agent, _last_user_text(context.messages), context)
        if name is None:
            await call_next()
            return
        if not context.stream:
            try:
                with _agent_scope(name):
                    await call_next()
            except BaseException as exc:
                flow._guard(flow._finish, context, "", exc)
                raise
            flow._guard(flow._finish, context, getattr(context.result, "text", "") or "")
            return

        # Streaming: the work happens while the caller iterates, so keep this
        # agent current around every pull and finish when the stream ends.
        with _agent_scope(name):
            await call_next()
        stream = context.result
        done = {"finished": False}

        def on_result(final: Any) -> None:
            done["finished"] = True
            flow._guard(flow._finish, context, getattr(final, "text", "") or "")

        def on_cleanup() -> None:
            if not done["finished"]:
                done["finished"] = True
                flow._guard(flow._finish, context, "")

        if stream is not None and hasattr(stream, "with_pull_context_manager"):
            context.result = (stream.with_pull_context_manager(lambda: _agent_scope(name))
                              .with_result_hook(on_result)
                              .with_cleanup_hook(on_cleanup))
        else:
            on_cleanup()


class _ChatMiddleware(ChatMiddleware):
    def __init__(self, flow: AgentFlow) -> None:
        self.flow = flow

    async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
        flow = self.flow
        agent = flow._guard(flow._model_call, context.client, context.options)
        await call_next()
        if agent is None:
            return
        result = context.result
        if context.stream and hasattr(result, "with_result_hook"):
            context.result = result.with_result_hook(lambda final: flow._guard(flow._model_response, agent, final))
        else:
            flow._guard(flow._model_response, agent, result)


class _FunctionMiddleware(FunctionMiddleware):
    def __init__(self, flow: AgentFlow) -> None:
        self.flow = flow

    async def process(self, context: Any, call_next: Callable[[], Any]) -> None:
        flow = self.flow
        tool_name = getattr(context.function, "name", "tool")
        started = flow._guard(flow._tool_started, tool_name, context.arguments)
        if started is None:
            await call_next()
            return
        agent, token = started
        error: Optional[BaseException] = None
        try:
            await call_next()
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                _FRAME.reset(token)
            except ValueError:
                pass
            flow._guard(flow._tool_finished, agent, tool_name, context.result, error)


# ─── Workflow structure ──────────────────────────────────────────────────────

def _structure(workflow: Any) -> Dict[str, Any]:
    executors = workflow.executors
    nodes: List[Dict[str, Any]] = [{"id": "__start__", "label": "__start__", "kind": "start"}]
    for executor_id, executor in executors.items():
        nested = getattr(executor, "workflow", None)
        kind = "subgraph" if nested is not None and hasattr(nested, "executors") else "node"
        nodes.append({"id": executor_id, "label": executor_id, "kind": kind})
    nodes.append({"id": "__end__", "label": "__end__", "kind": "end"})

    edges: List[Dict[str, Any]] = [{"source": "__start__", "target": workflow.start_executor_id, "conditional": False}]
    seen: Set[Tuple[str, str]] = set()
    has_outgoing: Set[str] = set()
    for group in workflow.edge_groups:
        if type(group).__name__ == "InternalEdgeGroup":
            continue
        routed = getattr(group, "selection_func", None) is not None or hasattr(group, "cases")
        for edge in group.edges:
            pair = (edge.source_id, edge.target_id)
            if pair in seen or edge.source_id.startswith("internal:"):
                continue
            seen.add(pair)
            has_outgoing.add(edge.source_id)
            edges.append({"source": edge.source_id, "target": edge.target_id,
                          "conditional": bool(getattr(edge, "has_condition", False) or routed)})
    for executor_id in executors:
        if executor_id not in has_outgoing:
            edges.append({"source": executor_id, "target": "__end__", "conditional": False})
    return {"nodes": nodes, "edges": edges}


__all__: Sequence[str] = ["AgentFlow"]
