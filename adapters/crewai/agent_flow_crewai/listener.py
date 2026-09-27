"""CrewAI event listener that writes Agent Flow events.

Agent Flow's generic event source tails a JSONL file where each line is
``{"time": <seconds since start>, "type": <event type>, "payload": {...}}``
(see ``extension/src/protocol.ts`` and ``web/hooks/simulation/``).

Mapping from CrewAI to Agent Flow:

- A ``Crew`` run is an agent whose shape is a node graph of its tasks: task
  order, async (parallel) tasks and ``context`` dependencies become edges,
  and each task execution a ``node_start`` / ``node_end``. The agents that
  execute tasks are its subagents.
- A ``Flow`` run is an agent whose graph is its methods: ``@start``,
  ``@listen`` (including ``and_`` / ``or_``) and ``@router`` label edges,
  with each method execution as a node. Crews and agents kicked off inside
  a method are the flow's subagents.
- An agent run started by another agent's tool (hierarchical managers and
  ``allow_delegation`` use "Delegate work to coworker") is a subagent of
  that agent.
- Tool usage, LLM calls (ReAct thoughts and final answers), model ids and
  token usage map to the existing event types.

Everything comes from CrewAI's event bus. Parents are resolved with the
``parent_event_id`` scope chain CrewAI attaches to every event. The bus runs
handlers on a thread pool, so events can arrive out of order; they are
buffered briefly and written in emission order.
"""

import atexit
import heapq
import itertools
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from crewai.events import crewai_event_bus
from crewai.events.base_events import BaseEvent

logger = logging.getLogger(__name__)

_MAX_CONTENT = 10_000
_MAX_ARGS = 200
# How long an event waits for earlier-emitted events that the bus's thread
# pool may still be delivering.
_REORDER_WINDOW_S = 0.25

_FINAL_ANSWER = re.compile(r"Final Answer\s*:\s*", re.IGNORECASE)
_ACTION = re.compile(r"\n?\s*Action\s*\d*\s*:", re.IGNORECASE)
_THOUGHT = re.compile(r"^\s*Thought\s*:\s*", re.IGNORECASE)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for attr in ("raw", "output", "result", "text"):
        inner = getattr(value, attr, None)
        if isinstance(inner, str) and inner:
            return inner
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _react_parts(response: str) -> Tuple[str, str]:
    """(thought, final answer) from a ReAct-style completion."""
    final = ""
    match = _FINAL_ANSWER.search(response)
    head = response
    if match:
        final = response[match.end():].strip()
        head = response[: match.start()]
    action = _ACTION.search(head)
    if action:
        head = head[: action.start()]
    thought = _THOUGHT.sub("", head).strip()
    if not match and not action:  # plain text answer
        return "", response.strip()
    return thought, final


def _all_event_types() -> List[type]:
    seen: Set[type] = set()
    stack = [BaseEvent]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return sorted(seen, key=lambda c: c.__name__)


class _GraphRun:
    """A crew or flow run: its node graph and which outputs have flowed on."""

    def __init__(self, name: str, preds: Dict[str, List[Tuple[str, Optional[str]]]], starts: Sequence[str]) -> None:
        self.name = name
        self.preds = preds  # node -> [(source node, router label or None)]
        self.starts = set(starts)
        self.completions: Dict[str, int] = {}
        self.consumed: Dict[Tuple[str, str], int] = {}
        self.node_step: Dict[str, int] = {}
        self.results: Dict[str, str] = {}
        self.last_completed: Optional[str] = None

    def sources(self, node: str, trigger: Optional[str]) -> List[str]:
        fresh = []
        for src, label in self.preds.get(node, []):
            if self.completions.get(src, 0) <= self.consumed.get((src, node), 0):
                continue
            if label is not None and self.results.get(src) != label:
                continue
            fresh.append(src)
        if fresh:
            return sorted(set(fresh))
        if trigger is not None:
            return [trigger]
        return ["__start__"]


class AgentFlowListener:
    """Write Agent Flow JSONL events for CrewAI crews, flows and agents.

    Usage::

        listener = AgentFlowListener("agent-flow.jsonl", truncate=True)
        crew.kickoff()          # or flow.kickoff(), agent.kickoff(...)
        listener.close()        # flush and stop listening (also runs at exit)

    It listens on CrewAI's global event bus, so everything that runs while
    it is open is recorded; one listener is one Agent Flow session.

    Args:
        path: JSONL file to append to. Defaults to ``$AGENT_FLOW_EVENT_LOG``,
            then ``agent-flow-events.jsonl`` in the current directory.
        truncate: Empty the file before the first event is written.
    """

    def __init__(self, path: Optional[str] = None, *, truncate: bool = False) -> None:
        self.path = path or os.environ.get("AGENT_FLOW_EVENT_LOG") or "agent-flow-events.jsonl"
        self._truncate = truncate
        self._start: Optional[float] = None
        self._warned = False

        # Reorder buffer, drained by a worker thread in emission order
        self._heap: List[Tuple[float, int, int, float, Any, Any]] = []
        self._tiebreak = itertools.count()
        self._cond = threading.Condition()
        self._closed = False

        # Processing state (worker thread only)
        self._parent: Dict[str, Optional[str]] = {}
        self._scope: Dict[str, Tuple[str, ...]] = {}  # event_id -> scope descriptor
        self._names: Dict[str, str] = {}  # start event id -> Agent Flow name
        self._holder: Set[str] = set()
        self._agent_parent: Dict[str, Optional[str]] = {}
        self._user_message_sent = False
        self._structure_sent: Set[str] = set()
        self._models: Dict[str, str] = {}
        self._runs: Dict[str, _GraphRun] = {}  # start event id -> graph run
        self._task_nodes: Dict[str, str] = {}  # str(task.id) -> node id
        self._method_events: Dict[str, str] = {}  # method finished event id -> method

        self._handlers: List[Tuple[type, Any]] = []
        for event_type in _all_event_types():
            handler = self._make_handler()
            crewai_event_bus.register_handler(event_type, handler)
            self._handlers.append((event_type, handler))
        self._worker = threading.Thread(target=self._drain, name="agent-flow-crewai", daemon=True)
        self._worker.start()
        atexit.register(self.close)

    # ─── Intake and ordering ─────────────────────────────────────────────────

    def _make_handler(self) -> Any:
        def handler(source: Any, event: Any) -> None:
            stamp = getattr(event, "timestamp", None)
            key = stamp.timestamp() if stamp is not None else time.time()
            with self._cond:
                if self._closed:
                    return
                heapq.heappush(self._heap, (key, getattr(event, "emission_sequence", 0) or 0,
                                            next(self._tiebreak), time.monotonic(), event, source))
                self._cond.notify()
        return handler

    def _drain(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._heap:
                        wait = self._heap[0][3] + _REORDER_WINDOW_S - time.monotonic()
                        if wait <= 0 or self._closed:
                            break
                        self._cond.wait(wait)
                    elif self._closed:
                        return
                    else:
                        self._cond.wait()
                *_, event, source = heapq.heappop(self._heap)
            self._safe_process(event, source)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until the bus has delivered everything and it has been written."""
        try:
            crewai_event_bus.flush(timeout=timeout)
        except Exception:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cond:
                if not self._heap:
                    break
            time.sleep(0.02)
        time.sleep(0.02)  # let the worker finish the event it popped

    def close(self) -> None:
        """Flush, then stop listening. Safe to call more than once."""
        if self._closed:
            return
        self.flush()
        for event_type, handler in self._handlers:
            try:
                crewai_event_bus.off(event_type, handler)
            except Exception:
                pass
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._worker.join(timeout=5)

    def _safe_process(self, event: Any, source: Any) -> None:
        try:
            self._process(event, source)
        except Exception:  # pragma: no cover - defensive
            if not self._warned:
                self._warned = True
                logger.exception("agent-flow: failed to record a CrewAI event; further errors are suppressed")

    # ─── Output ──────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        now = time.monotonic()
        if self._start is None:
            self._start = now
        record = {"time": round(now - self._start, 3), "type": event_type, "payload": payload}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        mode = "w" if self._truncate else "a"
        self._truncate = False
        with open(self.path, mode, encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    # ─── Scope resolution ────────────────────────────────────────────────────

    def _resolve(self, event_id: Optional[str], kinds: Sequence[str]) -> Optional[Tuple[str, ...]]:
        """Nearest enclosing scope of one of ``kinds``, walking parent links."""
        seen = 0
        while event_id and seen < 200:
            scope = self._scope.get(event_id)
            if scope is not None and scope[0] in kinds:
                return scope
            event_id = self._parent.get(event_id)
            seen += 1
        return None

    def _owner(self, event: Any) -> Optional[str]:
        """The agent an LLM / tool event belongs to."""
        scope = self._resolve(getattr(event, "parent_event_id", None), ("agent",))
        return scope[1] if scope else None

    # ─── Units (agents, crews, flows) ────────────────────────────────────────

    def _begin(self, start_id: str, base: str, task: str, parent_event_id: Optional[str]) -> str:
        name, n = base, 2
        while name in self._holder:
            name, n = f"{base} #{n}", n + 1
        self._holder.add(name)
        self._names[start_id] = name
        caller = self._resolve(parent_event_id, ("tool", "task", "method", "agent"))
        parent = caller[1] if caller else None
        self._agent_parent[name] = parent
        task = _truncate(task, _MAX_CONTENT)
        if parent is not None:
            self._emit("subagent_dispatch", {"parent": parent, "child": name, "task": task})
            self._emit("agent_spawn", {"name": name, "parent": parent, "task": task})
        else:
            self._emit("agent_spawn", {"name": name, "isMain": True, "task": task})
            if task and not self._user_message_sent:
                self._user_message_sent = True
                self._emit("message", {"agent": name, "role": "user", "content": task})
        return name

    def _finish(self, start_id: Optional[str], summary: str, error: Any = None) -> None:
        name = self._names.pop(start_id or "", None)
        if name is None:
            return
        if error is not None:
            self._emit("message", {"agent": name, "content": f"Error: {error}"})
            summary = summary or f"Error: {error}"
        parent = self._agent_parent.get(name)
        if parent is not None:
            self._emit("subagent_return", {"parent": parent, "child": name, "summary": _truncate(summary, _MAX_CONTENT)})
        self._emit("agent_complete", {"name": name})
        self._holder.discard(name)

    def _link_from_method(self, parent_event_id: Optional[str], child: str) -> None:
        """A crew / flow started inside a Flow method: make that method node open its graph."""
        caller = self._resolve(parent_event_id, ("tool", "task", "method", "agent"))
        if caller is not None and caller[0] == "method":
            node = {"id": caller[2], "label": caller[2], "kind": "subgraph", "child": child}
            self._emit("graph_structure", {"agent": caller[1], "nodes": [node], "edges": []})

    # ─── Graph runs (crews and flows) ────────────────────────────────────────

    def _node_start(self, run: _GraphRun, node: str, trigger: Optional[str]) -> None:
        sources = run.sources(node, trigger)
        for src in sources:
            if src != "__start__":
                run.consumed[(src, node)] = run.completions.get(src, 0)
        step = 1 + max((run.node_step.get(s, 0) for s in sources), default=0)
        run.node_step[node] = step
        self._emit("node_start", {"agent": run.name, "node": node, "step": step, "from": sources})

    def _node_end(self, run: _GraphRun, node: str, result: Any = None, error: Any = None) -> None:
        run.completions[node] = run.completions.get(node, 0) + 1
        run.last_completed = node
        if result is not None:
            run.results[node] = str(result)
        payload: Dict[str, Any] = {"agent": run.name, "node": node, "step": run.node_step.get(node, 0)}
        if error is not None:
            payload["error"] = _truncate(str(error), _MAX_CONTENT)
        self._emit("node_end", payload)

    def _graph_end(self, run: Optional[_GraphRun]) -> None:
        if run is None or not run.completions:
            return
        sinks = sorted(n for n, count in run.completions.items()
                       if not any(src == n and used == count for (src, _), used in run.consumed.items()))
        if not sinks and run.last_completed:
            sinks = [run.last_completed]
        step = max(run.node_step.values(), default=0) + 1
        self._emit("node_start", {"agent": run.name, "node": "__end__", "step": step, "from": sinks})

    # ─── Event dispatch ──────────────────────────────────────────────────────

    def _process(self, event: Any, source: Any) -> None:
        event_id = getattr(event, "event_id", None)
        if event_id:
            self._parent[event_id] = getattr(event, "parent_event_id", None)
        handler = getattr(self, f"_on_{getattr(event, 'type', '')}", None)
        if handler is not None:
            handler(event, source)

    # Crews

    def _on_crew_kickoff_started(self, event: Any, crew: Any) -> None:
        inputs = getattr(event, "inputs", None)
        first_task = (getattr(crew, "tasks", None) or [None])[0]
        task = _to_text(inputs) if inputs else getattr(first_task, "description", "") or ""
        name = self._begin(event.event_id, getattr(event, "crew_name", None) or getattr(crew, "name", None) or "crew",
                           task, event.parent_event_id)
        self._scope[event.event_id] = ("crew", name, event.event_id)
        self._link_from_method(event.parent_event_id, name)
        structure, preds, starts = _crew_structure(crew, self._task_nodes)
        self._runs[event.event_id] = _GraphRun(name, preds, starts)
        if name not in self._structure_sent and structure is not None:
            self._structure_sent.add(name)
            self._emit("graph_structure", {"agent": name, **structure})

    def _crew_end(self, event: Any, error: Any = None) -> None:
        start_id = getattr(event, "started_event_id", None)
        self._graph_end(self._runs.pop(start_id or "", None))
        self._finish(start_id, _to_text(getattr(event, "output", None)), error)

    def _on_crew_kickoff_completed(self, event: Any, source: Any) -> None:
        self._crew_end(event)

    def _on_crew_kickoff_failed(self, event: Any, source: Any) -> None:
        self._crew_end(event, getattr(event, "error", "failed"))

    def _task_run(self, event: Any) -> Optional[Tuple[_GraphRun, str]]:
        scope = self._resolve(event.parent_event_id, ("crew",))
        task = getattr(event, "task", None)
        if scope is None or task is None:
            return None
        run = self._runs.get(scope[2])
        node = self._task_nodes.get(str(getattr(task, "id", "")))
        return (run, node) if run is not None and node is not None else None

    def _on_task_started(self, event: Any, source: Any) -> None:
        found = self._task_run(event)
        if found is None:
            return
        run, node = found
        self._scope[event.event_id] = ("task", run.name, node)
        self._node_start(run, node, None)

    def _on_task_completed(self, event: Any, source: Any) -> None:
        found = self._task_run(event)
        if found is not None:
            self._node_end(found[0], found[1], _to_text(getattr(event, "output", None)))

    def _on_task_failed(self, event: Any, source: Any) -> None:
        found = self._task_run(event)
        if found is not None:
            self._node_end(found[0], found[1], error=getattr(event, "error", "failed"))

    # Agents

    def _agent_started(self, event: Any, role: str, task: str) -> None:
        name = self._begin(event.event_id, role or "agent", task, event.parent_event_id)
        self._scope[event.event_id] = ("agent", name)

    def _on_agent_execution_started(self, event: Any, source: Any) -> None:
        agent, task = getattr(event, "agent", None), getattr(event, "task", None)
        self._agent_started(event, getattr(agent, "role", "agent"),
                            getattr(task, "description", None) or getattr(event, "task_prompt", "") or "")

    def _on_agent_execution_completed(self, event: Any, source: Any) -> None:
        self._finish(event.started_event_id, _to_text(getattr(event, "output", "")))

    def _on_agent_execution_error(self, event: Any, source: Any) -> None:
        self._finish(event.started_event_id, "", getattr(event, "error", "failed"))

    def _on_lite_agent_execution_started(self, event: Any, source: Any) -> None:
        info = getattr(event, "agent_info", None) or {}
        messages = getattr(event, "messages", None)
        self._agent_started(event, info.get("role", "agent"), messages if isinstance(messages, str) else _to_text(messages))

    def _on_lite_agent_execution_completed(self, event: Any, source: Any) -> None:
        self._finish(event.started_event_id, _to_text(getattr(event, "output", "")))

    def _on_lite_agent_execution_error(self, event: Any, source: Any) -> None:
        self._finish(event.started_event_id, "", getattr(event, "error", "failed"))

    # Tools

    def _on_tool_usage_started(self, event: Any, source: Any) -> None:
        agent = self._owner(event)
        if agent is None:
            return
        args = getattr(event, "tool_args", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        tool_name = getattr(event, "tool_name", "tool")
        payload: Dict[str, Any] = {"agent": agent, "tool": tool_name, "args": _truncate(_to_text(args), _MAX_ARGS)}
        if isinstance(args, dict):
            payload["inputData"] = args
        self._emit("tool_call_start", payload)
        self._scope[event.event_id] = ("tool", agent, tool_name)

    def _tool_end(self, event: Any, result: str, error: Any = None) -> None:
        scope = self._scope.get(getattr(event, "started_event_id", None) or "")
        if scope is None or scope[0] != "tool":
            return
        payload: Dict[str, Any] = {"agent": scope[1], "tool": scope[2], "result": _truncate(result, _MAX_CONTENT)}
        if error is not None:
            payload["isError"] = True
            payload["errorMessage"] = payload["result"] = _truncate(str(error), _MAX_CONTENT)
        self._emit("tool_call_end", payload)

    def _on_tool_usage_finished(self, event: Any, source: Any) -> None:
        self._tool_end(event, _to_text(getattr(event, "output", "")))

    def _on_tool_usage_error(self, event: Any, source: Any) -> None:
        self._tool_end(event, "", getattr(event, "error", "failed"))

    # LLM calls

    def _on_llm_call_started(self, event: Any, source: Any) -> None:
        agent = self._owner(event)
        model = getattr(event, "model", None)
        if agent is not None and isinstance(model, str) and model and self._models.get(agent) != model:
            self._models[agent] = model
            self._emit("model_detected", {"agent": agent, "model": model})

    def _on_llm_call_completed(self, event: Any, source: Any) -> None:
        agent = self._owner(event)
        if agent is None:
            return
        response = getattr(event, "response", None)
        if isinstance(response, str):
            thought, final = _react_parts(response)
            if thought:
                self._emit("message", {"agent": agent, "role": "thinking", "content": _truncate(thought, _MAX_CONTENT)})
            if final:
                self._emit("message", {"agent": agent, "role": "assistant", "content": _truncate(final, _MAX_CONTENT)})
        usage = getattr(event, "usage", None) or {}
        tokens = usage.get("total_tokens") or ((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
        if tokens:
            self._emit("context_update", {"agent": agent, "tokens": tokens})

    def _on_llm_call_failed(self, event: Any, source: Any) -> None:
        agent = self._owner(event)
        if agent is not None:
            self._emit("message", {"agent": agent, "content": f"LLM error: {getattr(event, 'error', '')}"})

    # Flows

    def _is_user_flow(self, source: Any) -> bool:
        return type(source).__name__ != "AgentExecutor" and hasattr(source, "kickoff")

    def _on_flow_started(self, event: Any, flow: Any) -> None:
        if not self._is_user_flow(flow):
            return
        inputs = getattr(event, "inputs", None)
        name = self._begin(event.event_id, getattr(event, "flow_name", None) or type(flow).__name__,
                           _to_text(inputs) if inputs else "", event.parent_event_id)
        self._scope[event.event_id] = ("flow", name)
        self._link_from_method(event.parent_event_id, name)
        structure, preds, starts = _flow_structure(type(flow))
        self._runs[event.event_id] = _GraphRun(name, preds, starts)
        if name not in self._structure_sent and structure is not None:
            self._structure_sent.add(name)
            self._emit("graph_structure", {"agent": name, **structure})

    def _flow_end(self, event: Any, error: Any = None) -> None:
        start_id = getattr(event, "started_event_id", None)
        if start_id not in self._runs:
            return
        self._graph_end(self._runs.pop(start_id, None))
        self._finish(start_id, _to_text(getattr(event, "result", None)), error)

    def _on_flow_finished(self, event: Any, source: Any) -> None:
        self._flow_end(event)

    def _on_flow_failed(self, event: Any, source: Any) -> None:
        self._flow_end(event, getattr(event, "error", "failed"))

    def _flow_run(self, event: Any) -> Optional[_GraphRun]:
        flow_id = event.parent_event_id
        while flow_id and flow_id not in self._runs:
            flow_id = self._parent.get(flow_id)
        run = self._runs.get(flow_id or "")
        return run if run is not None and self._scope.get(flow_id or "", ("",))[0] == "flow" else None

    def _on_method_execution_started(self, event: Any, source: Any) -> None:
        run = self._flow_run(event)
        if run is None:
            return
        method = event.method_name
        trigger = self._method_events.get(getattr(event, "triggered_by_event_id", None) or "")
        self._scope[event.event_id] = ("method", run.name, method)
        self._node_start(run, method, trigger)

    def _on_method_execution_finished(self, event: Any, source: Any) -> None:
        run = self._flow_run(event)
        if run is None:
            return
        self._method_events[event.event_id] = event.method_name
        self._node_end(run, event.method_name, getattr(event, "result", None))

    def _on_method_execution_failed(self, event: Any, source: Any) -> None:
        run = self._flow_run(event)
        if run is not None:
            self._node_end(run, event.method_name, error=getattr(event, "error", "failed"))


# ─── Structures ──────────────────────────────────────────────────────────────

def _task_ids(tasks: Sequence[Any]) -> List[Tuple[Any, str, str]]:
    """(task, node id, label) with unique ids."""
    out, used = [], set()
    for i, task in enumerate(tasks):
        base = getattr(task, "name", None) or f"task {i + 1}"
        node, n = base, 2
        while node in used:
            node, n = f"{base} #{n}", n + 1
        used.add(node)
        label = getattr(task, "name", None) or _truncate((getattr(task, "description", "") or base).strip().split("\n")[0], 32)
        out.append((task, node, label))
    return out


def _crew_structure(crew: Any, task_nodes: Dict[str, str]):
    tasks = list(getattr(crew, "tasks", None) or [])
    if not tasks:
        return None, {}, []
    ids = _task_ids(tasks)
    by_task = {id(t): node for t, node, _ in ids}
    for task, node, _ in ids:
        task_nodes[str(getattr(task, "id", id(task)))] = node

    preds: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    last_sync: Optional[str] = None
    pending_async: List[str] = []
    starts: List[str] = []
    for task, node, _ in ids:
        context = getattr(task, "context", None)
        if isinstance(context, list) and context:
            sources = [by_task[id(c)] for c in context if id(c) in by_task]
        elif pending_async and not getattr(task, "async_execution", False):
            sources = list(pending_async)
        else:
            sources = [last_sync] if last_sync else []
        preds[node] = [(s, None) for s in sources]
        if not sources:
            starts.append(node)
        if getattr(task, "async_execution", False):
            pending_async.append(node)
        else:
            pending_async = []
            last_sync = node

    nodes = [{"id": "__start__", "label": "__start__", "kind": "start"}]
    nodes += [{"id": node, "label": label, "kind": "node"} for _, node, label in ids]
    nodes.append({"id": "__end__", "label": "__end__", "kind": "end"})
    edges = [{"source": "__start__", "target": s, "conditional": False} for s in starts]
    has_outgoing = set()
    for node, sources in preds.items():
        for src, _ in sources:
            has_outgoing.add(src)
            edges.append({"source": src, "target": node, "conditional": False})
    edges += [{"source": node, "target": "__end__", "conditional": False} for _, node, _ in ids if node not in has_outgoing]
    return {"nodes": nodes, "edges": edges}, preds, starts


def _flow_structure(flow_cls: type):
    try:
        from crewai.flow.visualization import build_flow_structure
        structure = build_flow_structure(flow_cls)
    except Exception:
        return None, {}, []
    methods = list(structure.get("nodes", {}).keys())
    starts = list(structure.get("start_methods") or [])
    preds: Dict[str, List[Tuple[str, Optional[str]]]] = {m: [] for m in methods}
    edges = [{"source": "__start__", "target": s, "conditional": False} for s in starts]
    has_outgoing: Set[str] = set()
    for edge in structure.get("edges", []):
        src, dst = edge.get("source"), edge.get("target")
        if src not in preds or dst not in preds:
            continue
        label = edge.get("router_event") if edge.get("is_router_event") else None
        preds[dst].append((src, label))
        has_outgoing.add(src)
        entry: Dict[str, Any] = {"source": src, "target": dst, "conditional": bool(edge.get("is_router_event"))}
        if label:
            entry["label"] = str(label)
        edges.append(entry)
    edges += [{"source": m, "target": "__end__", "conditional": False} for m in methods if m not in has_outgoing]
    nodes = [{"id": "__start__", "label": "__start__", "kind": "start"}]
    nodes += [{"id": m, "label": m, "kind": "node"} for m in methods]
    nodes.append({"id": "__end__", "label": "__end__", "kind": "end"})
    return {"nodes": nodes, "edges": edges}, preds, starts


__all__: Sequence[str] = ["AgentFlowListener"]
