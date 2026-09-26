"""Strands Agents hook provider that writes Agent Flow events.

Agent Flow's generic event source tails a JSONL file where each line is
``{"time": <seconds since start>, "type": <event type>, "payload": {...}}``
(see ``extension/src/protocol.ts`` and ``web/hooks/simulation/``).

Mapping from Strands to Agent Flow:

- Every ``Agent`` is an Agent Flow agent. The first one invoked with no
  caller is the main agent.
- An agent invoked from inside another agent's tool (the "agents as tools"
  pattern) is a subagent of that agent.
- A ``Graph`` or ``Swarm`` is an agent whose shape is a node graph: its
  nodes and edges are sent as ``graph_structure`` and each node execution
  as ``node_start`` / ``node_end``, so the Graph panel can draw the real
  routing, parallel batches and handoffs. The agents running in its nodes
  are its subagents, named after their node. Graphs nest.
- Tool calls, assistant text, reasoning, model ids and context size map to
  the existing event types.

Callers are found with context variables: Strands copies the context into
the threads and tasks it runs agents, tools and graph nodes in, so a nested
agent sees which tool or node it was started from.
"""

import contextvars
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)

logger = logging.getLogger(__name__)

_MAX_CONTENT = 10_000
_MAX_ARGS = 200


@dataclass(frozen=True)
class _Frame:
    """What is running right now in this context: a tool call or a graph node."""

    kind: str  # "tool" | "node"
    parent: str  # Agent Flow name of the agent / graph that owns it
    label: str  # tool name or node id


_CURRENT: "contextvars.ContextVar[Optional[_Frame]]" = contextvars.ContextVar("agent_flow_frame", default=None)


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


def _blocks_text(blocks: Any) -> str:
    """Text of a list of Strands content blocks (or a plain string)."""
    if isinstance(blocks, str):
        return blocks
    parts: List[str] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            parts.append(str(block["text"]))
        elif "json" in block:
            parts.append(_to_text(block["json"]))
    return "\n".join(p for p in parts if p)


def _messages_text(messages: Any) -> str:
    """Latest user text in a list of Strands messages."""
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            text = _blocks_text(message.get("content"))
            if text:
                return text
    return ""


class _RunState:
    """Per multi-agent run: which node outputs have not yet flowed anywhere."""

    def __init__(self) -> None:
        self.completions: Dict[str, int] = {}  # node -> times completed
        self.consumed: Dict[Tuple[str, str], int] = {}  # (src, dst) -> completion count consumed
        self.node_step: Dict[str, int] = {}
        self.last_completed: Optional[str] = None
        self.running: Dict[str, int] = {}  # node -> step of the in-flight execution
        self.task_sent = False


class AgentFlowHooks(HookProvider):
    """Write Agent Flow JSONL events for Strands agents, graphs and swarms.

    Usage::

        flow = AgentFlowHooks("agent-flow.jsonl", truncate=True)
        agent = Agent(tools=[...], hooks=[flow])
        # or attach to an existing agent / Graph / Swarm and everything inside it:
        flow.instrument(graph)

    One instance is one Agent Flow session; share it between all agents you
    want in the same view. Agents that are only reachable through your own
    tool functions (agents as tools) need ``hooks=[flow]`` or
    ``flow.instrument(agent)`` too.

    Args:
        path: JSONL file to append to. Defaults to ``$AGENT_FLOW_EVENT_LOG``,
            then ``agent-flow-events.jsonl`` in the current directory.
        truncate: Empty the file before the first event is written.
    """

    def __init__(self, path: Optional[str] = None, *, truncate: bool = False) -> None:
        self.path = path or os.environ.get("AGENT_FLOW_EVENT_LOG") or "agent-flow-events.jsonl"
        self._truncate = truncate
        self._lock = threading.RLock()
        self._start: Optional[float] = None
        self._warned = False

        # Names are held only while an agent runs: a later agent with the same
        # name reuses it, concurrent ones get "name #2", ...
        self._names: Dict[int, str] = {}  # id(running agent or multi-agent) -> Agent Flow name
        self._holder: Dict[str, int] = {}  # running name -> id of the object holding it
        self._parent: Dict[str, Optional[str]] = {}
        self._user_message_sent = False
        self._structure_sent: Set[str] = set()
        self._models: Dict[str, str] = {}
        self._context: Dict[str, int] = {}
        self._runs: Dict[str, _RunState] = {}
        self._tokens: Dict[str, contextvars.Token] = {}
        self._instrumented: Set[int] = set()

    # ─── Wiring ──────────────────────────────────────────────────────────────

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        self._instrumented.add(id(registry))
        for event_type, callback in (
            (BeforeInvocationEvent, self._on_agent_start),
            (AfterInvocationEvent, self._on_agent_end),
            (MessageAddedEvent, self._on_message),
            (BeforeModelCallEvent, self._on_model_call),
            (AfterModelCallEvent, self._on_model_done),
            (BeforeToolCallEvent, self._on_tool_start),
            (AfterToolCallEvent, self._on_tool_end),
            (BeforeMultiAgentInvocationEvent, self._on_multi_start),
            (AfterMultiAgentInvocationEvent, self._on_multi_end),
            (BeforeNodeCallEvent, self._on_node_start),
            (AfterNodeCallEvent, self._on_node_end),
        ):
            registry.add_callback(event_type, self._safe(callback))

    def instrument(self, target: Any) -> Any:
        """Attach these hooks to an Agent, Graph or Swarm and, recursively, to
        every agent and nested graph in its nodes. Returns ``target``."""
        registry = getattr(target, "hooks", None)
        if registry is not None and id(registry) not in self._instrumented and hasattr(registry, "add_hook"):
            registry.add_hook(self)
        nodes = getattr(target, "nodes", None)
        if isinstance(nodes, dict):
            for node in nodes.values():
                executor = getattr(node, "executor", None)
                if executor is not None:
                    self.instrument(executor)
        return target

    def _safe(self, callback: Any) -> Any:
        """Never let the visualizer break the agent run."""

        def wrapped(event: Any) -> None:
            try:
                callback(event)
            except Exception:  # pragma: no cover - defensive
                if not self._warned:
                    self._warned = True
                    logger.exception("agent-flow: hook failed; further errors are suppressed")

        return wrapped

    # ─── Output ──────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            now = time.monotonic()
            if self._start is None:
                self._start = now
            event = {"time": round(now - self._start, 3), "type": event_type, "payload": payload}
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            mode = "w" if self._truncate else "a"
            self._truncate = False
            with open(self.path, mode, encoding="utf-8") as f:
                f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")

    # ─── Agents ──────────────────────────────────────────────────────────────

    def _name_for(self, obj: Any, preferred: Optional[str]) -> str:
        """Agent Flow name for a starting agent / multi-agent, unique among running ones."""
        key = id(obj)
        if key in self._names:
            return self._names[key]
        base = preferred or getattr(obj, "name", None) or getattr(obj, "id", None) or type(obj).__name__
        if base in ("default_graph", "default_swarm"):
            base = type(obj).__name__.lower()
        name, n = str(base), 2
        while name in self._holder:
            name, n = f"{base} #{n}", n + 1
        self._holder[name] = key
        self._names[key] = name
        return name

    def _begin(self, obj: Any, task: str) -> str:
        """Start (or restart) an agent or multi-agent; returns its name."""
        frame = _CURRENT.get()
        preferred = frame.label if frame is not None and frame.kind == "node" else None
        name = self._name_for(obj, preferred)
        parent = frame.parent if frame is not None else None
        if parent == name:
            parent = None
        task = _truncate(task, _MAX_CONTENT)

        if parent is not None:
            self._parent[name] = parent
            self._emit("subagent_dispatch", {"parent": parent, "child": name, "task": task})
            self._emit("agent_spawn", {"name": name, "parent": parent, "task": task})
        else:
            self._parent[name] = None
            self._emit("agent_spawn", {"name": name, "isMain": True, "task": task})
            self._user_message(name, task)
        return name

    def _user_message(self, name: str, task: str) -> None:
        """The session's first prompt, shown on the main agent."""
        if task and not self._user_message_sent:
            self._user_message_sent = True
            self._emit("message", {"agent": name, "role": "user", "content": task})

    def _finish(self, obj: Any, name: str, summary: str) -> None:
        parent = self._parent.get(name)
        if parent is not None:
            self._emit("subagent_return", {"parent": parent, "child": name, "summary": _truncate(summary, _MAX_CONTENT)})
        self._emit("agent_complete", {"name": name})
        self._names.pop(id(obj), None)
        if self._holder.get(name) == id(obj):
            del self._holder[name]

    def _on_agent_start(self, event: BeforeInvocationEvent) -> None:
        with self._lock:
            self._begin(event.agent, _messages_text(event.messages))

    def _on_agent_end(self, event: AfterInvocationEvent) -> None:
        with self._lock:
            name = self._names.get(id(event.agent))
            if name is None:
                return
            self._context_update(event.agent, name)
            result = getattr(event, "result", None)
            self._finish(event.agent, name, str(result).strip() if result is not None else "")

    def _on_message(self, event: MessageAddedEvent) -> None:
        with self._lock:
            name = self._names.get(id(event.agent))
            message = event.message
            if name is None or message.get("role") != "assistant":
                return
            for block in message.get("content") or []:
                if "reasoningContent" in block:
                    reasoning = block["reasoningContent"].get("reasoningText", {}).get("text") or ""
                    if reasoning:
                        self._emit("message", {"agent": name, "role": "thinking", "content": _truncate(reasoning, _MAX_CONTENT)})
                elif "text" in block and str(block["text"]).strip():
                    self._emit("message", {"agent": name, "role": "assistant", "content": _truncate(str(block["text"]), _MAX_CONTENT)})

    def _on_model_call(self, event: BeforeModelCallEvent) -> None:
        with self._lock:
            name = self._names.get(id(event.agent))
            if name is None:
                return
            self._context_update(event.agent, name)
            model = None
            try:
                config = event.agent.model.get_config()
                model = config.get("model_id") or config.get("model") if isinstance(config, dict) else None
            except Exception:
                model = None
            if isinstance(model, str) and model and self._models.get(name) != model:
                self._models[name] = model
                self._emit("model_detected", {"agent": name, "model": model})

    def _on_model_done(self, event: AfterModelCallEvent) -> None:
        with self._lock:
            name = self._names.get(id(event.agent))
            if name is not None and event.exception is not None:
                self._emit("message", {"agent": name, "content": f"Model error: {event.exception!r}"})

    def _context_update(self, agent: Any, name: str) -> None:
        """Usage lands after each model reply, so report it at the next hook."""
        try:
            size = agent.event_loop_metrics.latest_context_size
        except Exception:
            size = None
        if isinstance(size, int) and size > 0 and self._context.get(name) != size:
            self._context[name] = size
            self._emit("context_update", {"agent": name, "tokens": size})

    # ─── Tools ───────────────────────────────────────────────────────────────

    def _on_tool_start(self, event: BeforeToolCallEvent) -> None:
        with self._lock:
            name = self._names.get(id(event.agent))
            if name is None:
                return
            self._context_update(event.agent, name)
            tool_use = event.tool_use
            tool_input = tool_use.get("input")
            payload: Dict[str, Any] = {
                "agent": name,
                "tool": tool_use.get("name", "tool"),
                "args": _truncate(_to_text(tool_input), _MAX_ARGS),
            }
            if isinstance(tool_input, dict):
                payload["inputData"] = tool_input
            self._emit("tool_call_start", payload)
            # Agents started inside this tool become this agent's subagents.
            token = _CURRENT.set(_Frame("tool", name, payload["tool"]))
            self._tokens[f"tool:{tool_use.get('toolUseId')}"] = token

    def _on_tool_end(self, event: AfterToolCallEvent) -> None:
        with self._lock:
            tool_use = event.tool_use
            self._restore(f"tool:{tool_use.get('toolUseId')}")
            name = self._names.get(id(event.agent))
            if name is None:
                return
            result = event.result or {}
            text = _blocks_text(result.get("content"))
            error = event.exception is not None or result.get("status") == "error"
            payload: Dict[str, Any] = {"agent": name, "tool": tool_use.get("name", "tool"), "result": _truncate(text, _MAX_CONTENT)}
            if error:
                payload["isError"] = True
                payload["errorMessage"] = _truncate(str(event.exception) if event.exception is not None else text, _MAX_CONTENT)
            self._emit("tool_call_end", payload)

    def _restore(self, key: str) -> None:
        token = self._tokens.pop(key, None)
        if token is None:
            return
        try:
            _CURRENT.reset(token)
        except ValueError:
            # Callback ran in a different context than its Before* twin; the
            # value set there was local to that context anyway.
            pass

    # ─── Graphs and swarms ───────────────────────────────────────────────────

    def _on_multi_start(self, event: BeforeMultiAgentInvocationEvent) -> None:
        with self._lock:
            source = event.source
            # The run's state (and its task) is only created after this hook;
            # the task is announced at the first node instead.
            name = self._begin(source, "")
            self._runs[name] = _RunState()
            if name not in self._structure_sent:
                self._structure_sent.add(name)
                structure = _structure(source)
                if structure is not None:
                    self._emit("graph_structure", {"agent": name, **structure})

    def _on_multi_end(self, event: AfterMultiAgentInvocationEvent) -> None:
        with self._lock:
            name = self._names.get(id(event.source))
            if name is None:
                return
            run = self._runs.pop(name, None)
            if run is not None and run.completions:
                # Nodes whose latest output nothing downstream picked up
                sinks = sorted(n for n, count in run.completions.items()
                               if not any(src == n and used == count for (src, _), used in run.consumed.items()))
                if not sinks and run.last_completed:
                    sinks = [run.last_completed]
                step = max(run.node_step.values(), default=0) + 1
                self._emit("node_start", {"agent": name, "node": "__end__", "step": step, "from": sinks})
            self._finish(event.source, name, _result_summary(event.source))

    def _on_node_start(self, event: BeforeNodeCallEvent) -> None:
        with self._lock:
            source = event.source
            name = self._names.get(id(source))
            if name is None:
                return
            run = self._runs.setdefault(name, _RunState())
            if not run.task_sent:
                run.task_sent = True
                self._announce_task(source, name)
            node = event.node_id
            sources = self._node_sources(source, run, node, event.invocation_state or {})
            for src in sources:
                if src != "__start__":
                    run.consumed[(src, node)] = run.completions.get(src, 0)
            step = 1 + max((run.node_step.get(s, 0) for s in sources), default=0)
            run.running[node] = step
            self._emit("node_start", {"agent": name, "node": node, "step": step, "from": sources})
            token = _CURRENT.set(_Frame("node", name, node))
            self._tokens[f"node:{name}:{node}"] = token

    def _on_node_end(self, event: AfterNodeCallEvent) -> None:
        with self._lock:
            source = event.source
            name = self._names.get(id(source))
            node = event.node_id
            self._restore(f"node:{name}:{node}")
            if name is None:
                return
            run = self._runs.setdefault(name, _RunState())
            step = run.running.pop(node, run.node_step.get(node, 0))
            run.node_step[node] = step
            run.completions[node] = run.completions.get(node, 0) + 1
            run.last_completed = node
            payload: Dict[str, Any] = {"agent": name, "node": node, "step": step}
            status = getattr(getattr(getattr(source, "nodes", {}).get(node), "execution_status", None), "value", None)
            if status == "failed":
                payload["error"] = "node failed"
            self._emit("node_end", payload)

    def _announce_task(self, source: Any, name: str) -> None:
        task = getattr(getattr(source, "state", None), "task", "")
        text = _truncate(task if isinstance(task, str) else _blocks_text(task), _MAX_CONTENT)
        if not text:
            return
        parent = self._parent.get(name)
        if parent is None:
            self._user_message(name, text)
        else:  # re-sending the spawn updates the subagent's task
            self._emit("agent_spawn", {"name": name, "parent": parent, "task": text})

    def _node_sources(self, source: Any, run: _RunState, node: str, invocation_state: Dict[str, Any]) -> List[str]:
        """Which nodes this execution was reached from."""
        edges = getattr(source, "edges", None)
        if edges is not None:  # Graph: static edges, possibly conditional
            fresh = []
            for edge in edges:
                src, dst = edge.from_node.node_id, edge.to_node.node_id
                if dst != node or run.completions.get(src, 0) <= run.consumed.get((src, dst), 0):
                    continue
                if edge.condition is not None:
                    try:
                        if not edge.should_traverse(source.state, invocation_state=invocation_state):
                            continue
                    except Exception:
                        pass
                fresh.append(src)
            if fresh:
                return sorted(fresh)
            return ["__start__"]
        # Swarm: whoever finished last handed off to this node
        if run.last_completed is not None and run.last_completed != node:
            return [run.last_completed]
        return ["__start__"] if not run.completions else [run.last_completed or "__start__"]


def _structure(source: Any) -> Optional[Dict[str, Any]]:
    nodes_attr = getattr(source, "nodes", None)
    if not isinstance(nodes_attr, dict):
        return None
    from strands.multiagent.base import MultiAgentBase

    nodes: List[Dict[str, Any]] = [{"id": "__start__", "label": "__start__", "kind": "start"}]
    for node_id, node in nodes_attr.items():
        kind = "subgraph" if isinstance(getattr(node, "executor", None), MultiAgentBase) else "node"
        nodes.append({"id": node_id, "label": node_id, "kind": kind})
    nodes.append({"id": "__end__", "label": "__end__", "kind": "end"})

    edges: List[Dict[str, Any]] = []
    graph_edges = getattr(source, "edges", None)
    if graph_edges is not None:  # Graph
        for entry in sorted(n.node_id for n in getattr(source, "entry_points", []) or []):
            edges.append({"source": "__start__", "target": entry, "conditional": False})
        with_outgoing = set()
        for edge in sorted(graph_edges, key=lambda e: (e.from_node.node_id, e.to_node.node_id)):
            with_outgoing.add(edge.from_node.node_id)
            edges.append({"source": edge.from_node.node_id, "target": edge.to_node.node_id,
                          "conditional": edge.condition is not None})
        for node_id in nodes_attr:
            if node_id not in with_outgoing:
                edges.append({"source": node_id, "target": "__end__", "conditional": False})
    else:  # Swarm: handoffs are dynamic, only the entry point is known
        entry = getattr(source, "entry_point", None)
        entry_id = getattr(entry, "name", None) if entry is not None else next(iter(nodes_attr), None)
        if entry_id in nodes_attr:
            edges.append({"source": "__start__", "target": entry_id, "conditional": False})
    return {"nodes": nodes, "edges": edges}


def _result_summary(source: Any) -> str:
    state = getattr(source, "state", None)
    results = getattr(state, "results", None)
    if isinstance(results, dict) and results:
        last = list(results.items())[-1]
        return f"{last[0]}: {str(getattr(last[1], 'result', last[1])).strip()}"
    return ""


__all__: Sequence[str] = ["AgentFlowHooks"]
