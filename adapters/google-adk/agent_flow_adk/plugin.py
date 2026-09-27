"""Google Agent Development Kit (ADK) plugin that writes Agent Flow events.

Agent Flow's generic event source tails a JSONL file where each line is
``{"time": <seconds since start>, "type": <event type>, "payload": {...}}``
(see ``extension/src/protocol.ts`` and ``web/hooks/simulation/``).

Mapping from ADK to Agent Flow:

- The runner's root agent is the main agent. Every agent that runs below it
  is a subagent of its parent agent (sub-agents, transfer targets and the
  children of workflow agents).
- ``SequentialAgent``, ``ParallelAgent`` and ``LoopAgent`` are agents whose
  shape is a node graph of their children: a chain, a fan-out, or a chain
  that loops back. Each child execution is a node; nested workflow agents
  are subgraph nodes that open their own graph.
- An agent run through ``AgentTool`` (agent as a tool) is a subagent of the
  agent that called the tool. ADK runs it in a nested runner with no link to
  the caller, so the link is carried in a context variable.
- Tool calls (including ``transfer_to_agent``), model text, thoughts, model
  names and token usage map to the existing event types.

Every callback returns ``None`` (so the run is never altered) and swallows
its own errors, since ADK aborts the run if a plugin raises.
"""

import contextvars
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from google.adk.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)

_MAX_CONTENT = 10_000
_MAX_ARGS = 200

# Agent Flow name of the agent whose AgentTool call is running in this task
_CALLER: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("agent_flow_adk_caller", default=None)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and set(value) == {"result"}:
        return _to_text(value["result"])
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _parts_text(content: Any) -> Tuple[str, str]:
    """(text, thoughts) from a google.genai Content."""
    texts: List[str] = []
    thoughts: List[str] = []
    for part in getattr(content, "parts", None) or []:
        text = getattr(part, "text", None)
        if not text:
            continue
        (thoughts if getattr(part, "thought", False) else texts).append(text)
    return "\n".join(texts), "\n".join(thoughts)


def _kind(agent: Any) -> str:
    """'sequential' | 'parallel' | 'loop' | 'agent'."""
    names = {c.__name__ for c in type(agent).__mro__}
    if "ParallelAgent" in names:
        return "parallel"
    if "LoopAgent" in names:
        return "loop"
    if "SequentialAgent" in names:
        return "sequential"
    return "agent"


def _structure(agent: Any) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Node graph of a workflow agent, and each child's static predecessors."""
    kind = _kind(agent)
    children = [c.name for c in getattr(agent, "sub_agents", None) or []]
    nodes = [{"id": "__start__", "label": "__start__", "kind": "start"}]
    for child in getattr(agent, "sub_agents", None) or []:
        nodes.append({"id": child.name, "label": child.name, "kind": "subgraph" if _kind(child) != "agent" else "node"})
    nodes.append({"id": "__end__", "label": "__end__", "kind": "end"})
    edges: List[Dict[str, Any]] = []
    preds: Dict[str, List[str]] = {c: [] for c in children}
    if not children:
        return {"nodes": nodes, "edges": edges}, preds
    if kind == "parallel":
        for c in children:
            edges.append({"source": "__start__", "target": c, "conditional": False})
            edges.append({"source": c, "target": "__end__", "conditional": False})
    else:
        edges.append({"source": "__start__", "target": children[0], "conditional": False})
        for a, b in zip(children, children[1:]):
            edges.append({"source": a, "target": b, "conditional": False})
            preds[b].append(a)
        if kind == "loop":
            # Loops back until a child escalates or max_iterations is reached
            edges.append({"source": children[-1], "target": children[0], "conditional": True, "label": "next iteration"})
            preds[children[0]].append(children[-1])
            edges.append({"source": children[-1], "target": "__end__", "conditional": True, "label": "done"})
            for c in children[:-1]:  # escalation can end the loop from any child
                edges.append({"source": c, "target": "__end__", "conditional": True, "label": "escalate"})
        else:
            edges.append({"source": children[-1], "target": "__end__", "conditional": False})
    return {"nodes": nodes, "edges": edges}, preds


class _GraphRun:
    def __init__(self, preds: Dict[str, List[str]]) -> None:
        self.preds = preds
        self.completions: Dict[str, int] = {}
        self.consumed: Dict[Tuple[str, str], int] = {}
        self.node_step: Dict[str, int] = {}
        self.running: Dict[str, int] = {}
        self.last_completed: Optional[str] = None

    def sources(self, node: str) -> List[str]:
        fresh = [s for s in self.preds.get(node, []) if self.completions.get(s, 0) > self.consumed.get((s, node), 0)]
        return sorted(fresh) or ["__start__"]


class _Unit:
    """A running agent."""

    def __init__(self, name: str, agent: Any, parent: Optional["_Unit"], invocation_id: str) -> None:
        self.name = name
        self.agent = agent
        self.parent = parent
        self.invocation_id = invocation_id
        self.graph: Optional[_GraphRun] = None
        self.node: Optional[str] = None  # its node id in the parent's graph, if any
        self.parent_name: Optional[str] = None  # who it reports back to (parent agent or AgentTool caller)
        self.last_text = ""


class AgentFlowPlugin(BasePlugin):
    """Write Agent Flow JSONL events for Google ADK runs.

    Usage::

        plugin = AgentFlowPlugin("agent-flow.jsonl", truncate=True)
        app = App(name="support", root_agent=root_agent, plugins=[plugin])
        runner = Runner(app=app, session_service=InMemorySessionService())

    One plugin instance is one Agent Flow session. Agents run through
    ``AgentTool`` are included automatically (ADK passes the plugins on).

    Args:
        path: JSONL file to append to. Defaults to ``$AGENT_FLOW_EVENT_LOG``,
            then ``agent-flow-events.jsonl`` in the current directory.
        truncate: Empty the file before the first event is written.
    """

    def __init__(self, path: Optional[str] = None, *, truncate: bool = False, name: str = "agent_flow") -> None:
        super().__init__(name=name)
        self.path = path or os.environ.get("AGENT_FLOW_EVENT_LOG") or "agent-flow-events.jsonl"
        self._truncate = truncate
        self._lock = threading.RLock()
        self._start: Optional[float] = None
        self._warned = False
        # (invocation id, agent name) -> running unit. Names, not object ids: agent
        # names are unique within an ADK tree, and ADK 2.x's node runtime runs
        # clones of the agents (so ``parent_agent`` is a different object).
        self._units: Dict[Tuple[str, str], _Unit] = {}
        self._holder: Set[str] = set()
        self._pending_user: Dict[str, str] = {}  # invocation id -> user message text
        self._user_message_sent = False
        self._structure_sent: Set[str] = set()
        self._models: Dict[str, str] = {}
        self._tools: Dict[str, Tuple[str, str, Optional[contextvars.Token]]] = {}  # call id -> (agent, tool, token)

    # ─── Output and safety ───────────────────────────────────────────────────

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

    def _guard(self, fn: Any, *args: Any) -> Any:
        try:
            with self._lock:
                return fn(*args)
        except Exception:  # pragma: no cover - defensive
            if not self._warned:
                self._warned = True
                logger.exception("agent-flow: failed to record an ADK callback; further errors are suppressed")
            return None

    # ─── Units ───────────────────────────────────────────────────────────────

    def _unit(self, invocation_id: str, agent_name: Optional[str]) -> Optional[_Unit]:
        return self._units.get((invocation_id, agent_name or ""))

    def _running_ancestor(self, invocation_id: str, agent: Any) -> Optional[_Unit]:
        parent = getattr(agent, "parent_agent", None)
        while parent is not None:
            unit = self._units.get((invocation_id, getattr(parent, "name", "")))
            if unit is not None:
                return unit
            parent = getattr(parent, "parent_agent", None)
        return None

    def _agent_started(self, agent: Any, callback_context: Any) -> None:
        invocation_id = callback_context.invocation_id
        key = (invocation_id, getattr(agent, "name", "") or "")
        if key in self._units:
            return
        base = getattr(agent, "name", None) or type(agent).__name__
        name, n = base, 2
        while name in self._holder:
            name, n = f"{base} #{n}", n + 1
        self._holder.add(name)

        parent = self._running_ancestor(invocation_id, agent)
        caller = _CALLER.get() if parent is None else None
        unit = _Unit(name, agent, parent, invocation_id)
        self._units[key] = unit
        task = _truncate(getattr(agent, "description", None) or "", _MAX_CONTENT)

        if parent is not None or caller is not None:
            parent_name = parent.name if parent is not None else caller
            if caller is not None and parent is None:
                task = self._pending_user.get(invocation_id) or task
            self._emit("subagent_dispatch", {"parent": parent_name, "child": name, "task": task})
            self._emit("agent_spawn", {"name": name, "parent": parent_name, "task": task})
            unit_parent_name = parent_name
        else:
            prompt = self._pending_user.get(invocation_id, "")
            self._emit("agent_spawn", {"name": name, "isMain": True, "task": _truncate(prompt, _MAX_CONTENT)})
            if prompt and not self._user_message_sent:
                self._user_message_sent = True
                self._emit("message", {"agent": name, "role": "user", "content": _truncate(prompt, _MAX_CONTENT)})
            unit_parent_name = None
        unit.parent_name = unit_parent_name

        # A child of a workflow agent is a node in its parent's graph
        if parent is not None and parent.graph is not None:
            run = parent.graph
            node = base
            sources = run.sources(node)
            for src in sources:
                if src != "__start__":
                    run.consumed[(src, node)] = run.completions.get(src, 0)
            step = 1 + max((run.node_step.get(s, 0) for s in sources), default=0)
            run.node_step[node] = step
            run.running[node] = step
            unit.node = node
            self._emit("node_start", {"agent": parent.name, "node": node, "step": step, "from": sources})

        # A workflow agent draws its children as a graph
        if _kind(agent) != "agent":
            structure, preds = _structure(agent)
            unit.graph = _GraphRun(preds)
            if name not in self._structure_sent:
                self._structure_sent.add(name)
                self._emit("graph_structure", {"agent": name, **structure})

    def _agent_finished(self, unit: _Unit, error: Any = None) -> None:
        key = (unit.invocation_id, getattr(unit.agent, "name", "") or "")
        if self._units.get(key) is not unit:
            return
        # Close anything still open below it first (e.g. transfer targets)
        for other in [u for u in self._units.values() if u.parent is unit]:
            self._agent_finished(other)
        del self._units[key]
        if unit.graph is not None and unit.graph.completions:
            run = unit.graph
            sinks = sorted(n for n, count in run.completions.items()
                           if not any(src == n and used == count for (src, _), used in run.consumed.items()))
            if not sinks and run.last_completed:
                sinks = [run.last_completed]
            step = max(run.node_step.values(), default=0) + 1
            self._emit("node_start", {"agent": unit.name, "node": "__end__", "step": step, "from": sinks})
        if unit.parent is not None and unit.parent.graph is not None and unit.node is not None:
            run = unit.parent.graph
            run.completions[unit.node] = run.completions.get(unit.node, 0) + 1
            run.last_completed = unit.node
            payload: Dict[str, Any] = {"agent": unit.parent.name, "node": unit.node,
                                       "step": run.running.pop(unit.node, run.node_step.get(unit.node, 0))}
            if error is not None:
                payload["error"] = _truncate(str(error), _MAX_CONTENT)
            self._emit("node_end", payload)
        if error is not None:
            self._emit("message", {"agent": unit.name, "content": f"Error: {error}"})
        parent_name = unit.parent_name
        if parent_name is not None:
            summary = unit.last_text or (f"Error: {error}" if error is not None else "")
            self._emit("subagent_return", {"parent": parent_name, "child": unit.name, "summary": _truncate(summary, _MAX_CONTENT)})
        self._emit("agent_complete", {"name": unit.name})
        self._holder.discard(unit.name)

    def _close_invocation(self, invocation_id: str, error: Any = None) -> None:
        roots = [u for (inv, _), u in list(self._units.items()) if inv == invocation_id and (u.parent is None or u.parent.invocation_id != invocation_id)]
        for unit in roots:
            self._agent_finished(unit, error)
        self._pending_user.pop(invocation_id, None)

    # ─── Callbacks ───────────────────────────────────────────────────────────

    async def on_user_message_callback(self, *, invocation_context: Any, user_message: Any) -> None:
        self._guard(lambda: self._pending_user.__setitem__(invocation_context.invocation_id, _parts_text(user_message)[0]))
        return None

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        self._guard(self._close_invocation, invocation_context.invocation_id)
        return None

    async def on_run_error_callback(self, *, invocation_context: Any, error: Exception) -> None:
        self._guard(self._close_invocation, invocation_context.invocation_id, error)
        return None

    async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        self._guard(self._agent_started, agent, callback_context)
        return None

    async def after_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        def finish() -> None:
            unit = self._units.get((callback_context.invocation_id, getattr(agent, "name", "") or ""))
            if unit is not None:
                self._agent_finished(unit)
        self._guard(finish)
        return None

    async def on_agent_error_callback(self, *, agent: Any, callback_context: Any, error: Exception) -> None:
        def finish() -> None:
            unit = self._units.get((callback_context.invocation_id, getattr(agent, "name", "") or ""))
            if unit is not None:
                self._agent_finished(unit, error)
        self._guard(finish)
        return None

    async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> None:
        def record() -> None:
            unit = self._unit(callback_context.invocation_id, callback_context.agent_name)
            if unit is None:
                return
            model = getattr(llm_request, "model", None)
            if not isinstance(model, str) or not model:
                model_obj = getattr(unit.agent, "model", None)
                model = model_obj if isinstance(model_obj, str) else getattr(model_obj, "model", None)
            if isinstance(model, str) and model and self._models.get(unit.name) != model:
                self._models[unit.name] = model
                self._emit("model_detected", {"agent": unit.name, "model": model})
        self._guard(record)
        return None

    async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
        def record() -> None:
            if getattr(llm_response, "partial", False):
                return
            unit = self._unit(callback_context.invocation_id, callback_context.agent_name)
            if unit is None:
                return
            text, thoughts = _parts_text(getattr(llm_response, "content", None))
            if thoughts:
                self._emit("message", {"agent": unit.name, "role": "thinking", "content": _truncate(thoughts, _MAX_CONTENT)})
            if text.strip():
                unit.last_text = text
                self._emit("message", {"agent": unit.name, "role": "assistant", "content": _truncate(text, _MAX_CONTENT)})
            usage = getattr(llm_response, "usage_metadata", None)
            if usage is not None:
                tokens = (getattr(usage, "prompt_token_count", 0) or 0) + (getattr(usage, "candidates_token_count", 0) or 0)
                if tokens:
                    self._emit("context_update", {"agent": unit.name, "tokens": tokens})
        self._guard(record)
        return None

    async def on_model_error_callback(self, *, callback_context: Any, llm_request: Any, error: Exception) -> None:
        def record() -> None:
            unit = self._unit(callback_context.invocation_id, callback_context.agent_name)
            if unit is not None:
                self._emit("message", {"agent": unit.name, "content": f"Model error: {error}"})
        self._guard(record)
        return None

    async def before_tool_callback(self, *, tool: Any, tool_args: Dict[str, Any], tool_context: Any) -> None:
        def record() -> None:
            unit = self._unit(tool_context.invocation_id, tool_context.agent_name)
            if unit is None:
                return
            name = getattr(tool, "name", None) or type(tool).__name__
            payload: Dict[str, Any] = {"agent": unit.name, "tool": name, "args": _truncate(_to_text(tool_args), _MAX_ARGS)}
            if isinstance(tool_args, dict):
                payload["inputData"] = tool_args
            self._emit("tool_call_start", payload)
            token = None
            if type(tool).__name__ == "AgentTool" or hasattr(tool, "agent") and hasattr(getattr(tool, "agent"), "sub_agents"):
                # The nested runner the tool starts should know who called it
                token = _CALLER.set(unit.name)
            self._tools[tool_context.function_call_id or f"{unit.name}:{name}"] = (unit.name, name, token)
        self._guard(record)
        return None

    def _tool_finished(self, tool: Any, tool_context: Any, result: Any, error: Any) -> None:
        name = getattr(tool, "name", None) or type(tool).__name__
        key = tool_context.function_call_id or ""
        entry = self._tools.pop(key, None)
        if entry is None:
            unit = self._unit(tool_context.invocation_id, tool_context.agent_name)
            entry = self._tools.pop(f"{unit.name}:{name}", None) if unit else None
        if entry is None:
            return
        agent, tool_name, token = entry
        if token is not None:
            try:
                _CALLER.reset(token)
            except ValueError:
                pass
        payload: Dict[str, Any] = {"agent": agent, "tool": tool_name, "result": _truncate(_to_text(result), _MAX_CONTENT)}
        if error is None and isinstance(result, dict) and result.get("status") == "error":
            # ADK's convention: tools report failures as {"status": "error", "error_message": ...}
            error = result.get("error_message") or result.get("error") or "error"
        if error is not None:
            payload["isError"] = True
            payload["errorMessage"] = payload["result"] = _truncate(str(error) or type(error).__name__, _MAX_CONTENT)
        self._emit("tool_call_end", payload)

    async def after_tool_callback(self, *, tool: Any, tool_args: Dict[str, Any], tool_context: Any, result: Any) -> None:
        self._guard(self._tool_finished, tool, tool_context, result, None)
        return None

    async def on_tool_error_callback(self, *, tool: Any, tool_args: Dict[str, Any], tool_context: Any, error: Exception) -> None:
        self._guard(self._tool_finished, tool, tool_context, None, error)
        return None


__all__: Sequence[str] = ["AgentFlowPlugin"]
