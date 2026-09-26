"""LangChain callback handler that writes Agent Flow events for LangGraph runs.

Agent Flow's generic event source tails a JSONL file where each line is
``{"time": <seconds since start>, "type": <event type>, "payload": {...}}``
(see ``extension/src/protocol.ts`` and ``web/hooks/simulation/``).

Mapping from LangGraph to Agent Flow:

- The top-level graph run is the main agent.
- A subgraph run (any compiled graph running inside a node, including
  ``create_react_agent`` agents) is a subagent named after the node that runs
  it. Nesting is preserved: a subgraph inside a subgraph becomes a
  grandchild.
- Plain function nodes are not agents. LLM and tool calls made inside them
  are attributed to the enclosing graph's agent.

Graph shape: pass the compiled graph (``graph=app``) to also emit each
graph's static structure (``graph_structure``: nodes plus normal and
conditional edges, one per graph including subgraphs). Every node execution
emits ``node_start`` (with the superstep and the nodes it was reached from)
and ``node_end``, so the UI can draw the real routing, loops and merges.
Without ``graph=``, the shape is still built from the observed hops.

Agents are identified with the ``langgraph_checkpoint_ns`` metadata that
LangGraph attaches to every callback. It is a ``|``-separated path of
``node:task_id`` segments, and the last segment is the node currently
running. Everything before it identifies the graph (the agent) the node
belongs to.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

AgentKey = Tuple[str, ...]

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


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", message)


def _message_role(message: Any) -> Optional[str]:
    if isinstance(message, dict):
        return message.get("role") or message.get("type")
    if isinstance(message, (list, tuple)) and len(message) == 2:
        return str(message[0])
    return getattr(message, "type", None)


def _split_content(content: Any) -> Tuple[str, str]:
    """Split message content into (text, thinking) strings."""
    if isinstance(content, str):
        return content, ""
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            text_parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(block.get("text", ""))
        elif kind == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif kind == "reasoning":
            summary = block.get("summary")
            if isinstance(summary, list):
                thinking_parts.extend(s.get("text", "") for s in summary if isinstance(s, dict))
            elif block.get("reasoning"):
                thinking_parts.append(str(block["reasoning"]))
    return "".join(text_parts), "\n".join(p for p in thinking_parts if p)


def _describe_input(inputs: Any) -> str:
    """Best-effort human-readable summary of a graph or node input."""
    if isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
        messages = inputs["messages"]
        for message in reversed(messages):
            if _message_role(message) in ("human", "user"):
                return _split_content(_message_content(message))[0]
        if messages:
            return _split_content(_message_content(messages[-1]))[0]
    return _to_text(inputs)


def _graph_path(key: AgentKey) -> str:
    """Namespace key -> node-name path, as used by ``get_subgraphs`` ("team|analyst")."""
    return "|".join(seg.split(":", 1)[0] for seg in key)


def _structure(graph: Any) -> Optional[Dict[str, Any]]:
    try:
        drawable = graph.get_graph()
    except Exception:
        return None
    nodes = []
    for node_id, node in drawable.nodes.items():
        kind = "start" if node_id == "__start__" else "end" if node_id == "__end__" else "node"
        nodes.append({"id": node_id, "label": getattr(node, "name", None) or node_id, "kind": kind})
    edges = []
    for edge in drawable.edges:
        entry: Dict[str, Any] = {"source": edge.source, "target": edge.target, "conditional": bool(edge.conditional)}
        if edge.data:
            entry["label"] = str(edge.data)
        edges.append(entry)
    return {"nodes": nodes, "edges": edges}


def _collect_structures(graph: Any) -> Dict[str, Dict[str, Any]]:
    """Static structure of a compiled graph and all its subgraphs, keyed by
    node-name path ("" is the top-level graph)."""
    structures: Dict[str, Dict[str, Any]] = {}
    root = _structure(graph)
    if root is None:
        return structures
    structures[""] = root
    try:
        try:
            subgraphs = list(graph.get_subgraphs(recurse=True))
        except TypeError:
            subgraphs = list(graph.get_subgraphs())
    except Exception:
        subgraphs = []
    for path, subgraph in subgraphs:
        sub = _structure(subgraph)
        if sub is not None:
            structures[path] = sub
    for path, structure in structures.items():
        for node in structure["nodes"]:
            if (f"{path}|{node['id']}" if path else node["id"]) in structures:
                node["kind"] = "subgraph"
    return structures


def _parse_ns(metadata: Optional[Dict[str, Any]]) -> Optional[AgentKey]:
    ns = (metadata or {}).get("langgraph_checkpoint_ns")
    if not isinstance(ns, str) or not ns:
        return None
    return tuple(ns.split("|"))


class AgentFlowCallbackHandler(BaseCallbackHandler):
    """Write Agent Flow JSONL events for LangGraph (and LangChain) runs.

    Usage::

        handler = AgentFlowCallbackHandler("agent-flow.jsonl")
        graph.invoke(inputs, config={"callbacks": [handler]})

    One handler instance is one Agent Flow session. Reuse it across
    invocations to keep adding to the same visualization, or create a new one
    (with ``truncate=True``) to start fresh.

    Args:
        path: JSONL file to append to. Defaults to ``$AGENT_FLOW_EVENT_LOG``,
            then ``agent-flow-events.jsonl`` in the current directory.
        main_agent_name: Name of the main agent. Defaults to the top-level
            graph's name.
        truncate: Empty the file before the first event is written.
        graph: The compiled graph being run. Optional; when given, its static
            structure (and its subgraphs') is sent so the UI can draw routes
            that weren't taken too.
    """

    raise_error = False

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        main_agent_name: Optional[str] = None,
        truncate: bool = False,
        graph: Any = None,
    ) -> None:
        self.path = path or os.environ.get("AGENT_FLOW_EVENT_LOG") or "agent-flow-events.jsonl"
        self._main_name_override = main_agent_name
        self._main_name: Optional[str] = None
        self._truncate = truncate
        self._lock = threading.RLock()
        self._start: Optional[float] = None

        # Agents spawned so far, keyed by namespace prefix ( () is the main agent).
        self._agents: Dict[AgentKey, str] = {}
        self._agent_parent: Dict[str, str] = {}
        self._active_names: Dict[str, AgentKey] = {}
        # Chain runs that own a namespace (the node run a subgraph executes in).
        self._ns_owner: Dict[AgentKey, UUID] = {}
        self._ns_input: Dict[AgentKey, Any] = {}
        self._owner_ns: Dict[UUID, AgentKey] = {}
        self._root_run: Optional[UUID] = None
        self._run_agent: Dict[UUID, str] = {}
        self._tool_names: Dict[UUID, str] = {}
        self._models: Dict[str, str] = {}
        # Graph shape: static structures by node-name path, and per running
        # graph (keyed by namespace prefix) the nodes executed at each step.
        self._structures = _collect_structures(graph) if graph is not None else {}
        self._structure_sent: set = set()
        self._graph_steps: Dict[AgentKey, Dict[int, List[str]]] = {}
        self._node_runs: Dict[UUID, Tuple[AgentKey, str, int]] = {}

    # ─── Output ──────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            now = time.monotonic()
            if self._start is None:
                self._start = now
            event = {"time": round(now - self._start, 3), "type": event_type, "payload": payload}
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
            mode = "w" if self._truncate else "a"
            self._truncate = False
            with open(self.path, mode, encoding="utf-8") as f:
                f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")

    # ─── Agent resolution ────────────────────────────────────────────────────

    def _ensure_main(self, name: Optional[str] = None, task: str = "") -> str:
        if () not in self._agents:
            self._main_name = self._main_name_override or name or "LangGraph"
            self._agents[()] = self._main_name
            self._active_names[self._main_name] = ()
            self._emit("agent_spawn", {"name": self._main_name, "isMain": True, "task": task})
            self._emit_structure((), self._main_name)
        return self._agents[()]

    def _agent_for(self, key: AgentKey) -> str:
        """Return the agent that owns a namespace prefix, spawning it (and its
        ancestors) on first sight."""
        if key in self._agents:
            return self._agents[key]
        if not key:
            return self._ensure_main()

        parent = self._agent_for(key[:-1])
        base = key[-1].split(":", 1)[0]
        # Parallel runs of the same subgraph (e.g. Send fan-out) get distinct names.
        name, n = base, 2
        while name in self._active_names and self._active_names[name] != key:
            name, n = f"{base} #{n}", n + 1

        task = _truncate(_describe_input(self._ns_input.get(key)), _MAX_CONTENT)
        self._agents[key] = name
        self._agent_parent[name] = parent
        self._active_names[name] = key
        self._emit("subagent_dispatch", {"parent": parent, "child": name, "task": task})
        self._emit("agent_spawn", {"name": name, "parent": parent, "task": task})
        self._emit_structure(key, name)
        return name

    def _agent_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        key = _parse_ns(metadata)
        if key is None:
            return self._ensure_main()
        return self._agent_for(key[:-1])

    # ─── Graph shape ─────────────────────────────────────────────────────────

    def _emit_structure(self, key: AgentKey, agent: str) -> None:
        structure = self._structures.get(_graph_path(key))
        if structure is not None and agent not in self._structure_sent:
            self._structure_sent.add(agent)
            self._emit("graph_structure", {"agent": agent, **structure})

    def _predecessors(self, graph_key: AgentKey, node: str, candidates: List[str]) -> List[str]:
        """Narrow the previous step's nodes to those with a static edge into
        ``node``. Keeps all of them when nothing matches (e.g. a Send)."""
        structure = self._structures.get(_graph_path(graph_key))
        if structure is None:
            return candidates
        sources = {e["source"] for e in structure["edges"] if e["target"] == node}
        matched = [c for c in candidates if c in sources]
        return matched or candidates

    def _node_started(self, run_id: UUID, key: AgentKey, metadata: Optional[Dict[str, Any]]) -> None:
        graph_key, node = key[:-1], key[-1].split(":", 1)[0]
        agent = self._agent_for(graph_key)
        steps = self._graph_steps.setdefault(graph_key, {})
        step = (metadata or {}).get("langgraph_step")
        if not isinstance(step, int):
            step = max(steps, default=0) + 1
        earlier = [s for s in steps if s < step]
        sources = self._predecessors(graph_key, node, list(steps[max(earlier)]) if earlier else ["__start__"])
        nodes_at_step = steps.setdefault(step, [])
        if node not in nodes_at_step:
            nodes_at_step.append(node)
        self._node_runs[run_id] = (graph_key, node, step)
        self._emit("node_start", {"agent": agent, "node": node, "step": step, "from": sources})

    def _graph_finished(self, graph_key: AgentKey, agent: str) -> None:
        """Record the final hop into ``__end__`` from the last step's nodes."""
        steps = self._graph_steps.pop(graph_key, None)
        if not steps:
            return
        last = max(steps)
        sources = self._predecessors(graph_key, "__end__", list(steps[last]))
        self._emit("node_start", {"agent": agent, "node": "__end__", "step": last + 1, "from": sources})

    # ─── Chains (graphs and nodes) ───────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if parent_run_id is None:
                self._root_run = run_id
                name = kwargs.get("name") or (serialized or {}).get("name")
                content = _truncate(_describe_input(inputs), _MAX_CONTENT)
                repeat = () in self._agents
                main = self._ensure_main(name, content)
                if repeat:
                    # Reactivate the main agent for a repeat invocation.
                    self._emit("agent_spawn", {"name": main, "isMain": True, "task": content})
                self._active_names[main] = ()
                if content:
                    self._emit("message", {"agent": main, "role": "user", "content": content})
                return

            key = _parse_ns(metadata)
            if key is not None and key not in self._ns_owner:
                # Outermost run for this namespace: the node's own run. If a
                # subgraph runs inside it, that subgraph's agent lives until
                # this run ends.
                self._ns_owner[key] = run_id
                self._ns_input[key] = inputs
                self._owner_ns[run_id] = key
                self._node_started(run_id, key, metadata)

    def _finish_run(self, run_id: UUID, outputs: Any, error: Optional[BaseException]) -> None:
        with self._lock:
            if run_id == self._root_run:
                self._root_run = None
                main = self._agents.get(())
                if main is None:
                    return
                if error is not None:
                    self._emit("message", {"agent": main, "content": f"Error: {error!r}"})
                self._graph_finished((), main)
                self._emit("agent_complete", {"name": main})
                self._active_names.pop(main, None)
                self._reset_namespaces()
                return

            key = self._owner_ns.pop(run_id, None)
            if key is None:
                return
            self._ns_owner.pop(key, None)
            self._ns_input.pop(key, None)
            node_run = self._node_runs.pop(run_id, None)
            name = self._agents.pop(key, None)
            if name is not None:
                # This node ran a subgraph: close out its agent first.
                self._graph_finished(key, name)
                parent = self._agent_parent.get(name, self._main_name or "")
                summary = f"Error: {error!r}" if error is not None else _describe_output(outputs)
                self._emit("subagent_return", {"parent": parent, "child": name, "summary": _truncate(summary, _MAX_CONTENT)})
                self._emit("agent_complete", {"name": name})
                if self._active_names.get(name) == key:
                    del self._active_names[name]
            if node_run is not None:
                graph_key, node, step = node_run
                agent = self._agents.get(graph_key)
                if agent is not None:
                    payload: Dict[str, Any] = {"agent": agent, "node": node, "step": step}
                    if error is not None:
                        payload["error"] = _truncate(str(error), _MAX_CONTENT)
                    self._emit("node_end", payload)

    def _reset_namespaces(self) -> None:
        main = self._agents.get(())
        self._agents = {(): main} if main is not None else {}
        self._ns_owner.clear()
        self._ns_input.clear()
        self._owner_ns.clear()
        self._active_names = {}
        self._graph_steps.clear()
        self._node_runs.clear()

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish_run(run_id, outputs, None)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish_run(run_id, None, error)

    # ─── LLM calls ───────────────────────────────────────────────────────────

    def _llm_start(self, run_id: UUID, metadata: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> None:
        with self._lock:
            agent = self._agent_from_metadata(metadata)
            self._run_agent[run_id] = agent
            params = kwargs.get("invocation_params") or {}
            model = (metadata or {}).get("ls_model_name") or params.get("model") or params.get("model_name")
            if isinstance(model, str) and model and self._models.get(agent) != model:
                self._models[agent] = model
                self._emit("model_detected", {"agent": agent, "model": model})

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._llm_start(run_id, metadata, kwargs)

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._llm_start(run_id, metadata, kwargs)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            agent = self._run_agent.pop(run_id, None)
            if agent is None:
                return
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    content = message.content if message is not None else generation.text
                    text, thinking = _split_content(content)
                    if thinking:
                        self._emit("message", {"agent": agent, "role": "thinking", "content": _truncate(thinking, _MAX_CONTENT)})
                    if text.strip():
                        self._emit("message", {"agent": agent, "role": "assistant", "content": _truncate(text, _MAX_CONTENT)})
                    usage = getattr(message, "usage_metadata", None) or {}
                    tokens = usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
                    if tokens:
                        self._emit("context_update", {"agent": agent, "tokens": tokens})

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            agent = self._run_agent.pop(run_id, None)
            if agent is not None:
                self._emit("message", {"agent": agent, "content": f"LLM error: {error!r}"})

    # ─── Tool calls ──────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            agent = self._agent_from_metadata(metadata)
            tool = kwargs.get("name") or (serialized or {}).get("name") or "tool"
            self._run_agent[run_id] = agent
            self._tool_names[run_id] = tool
            payload: Dict[str, Any] = {
                "agent": agent,
                "tool": tool,
                "args": _truncate(_to_text(inputs) if inputs is not None else input_str, _MAX_ARGS),
            }
            if isinstance(inputs, dict):
                payload["inputData"] = inputs
            self._emit("tool_call_start", payload)

    def _tool_end(self, run_id: UUID, result: str, error: Optional[BaseException]) -> None:
        with self._lock:
            agent = self._run_agent.pop(run_id, None)
            tool = self._tool_names.pop(run_id, None)
            if agent is None or tool is None:
                return
            payload: Dict[str, Any] = {"agent": agent, "tool": tool, "result": _truncate(result, _MAX_CONTENT)}
            if error is not None:
                payload["isError"] = True
                payload["errorMessage"] = _truncate(str(error), _MAX_CONTENT)
            self._emit("tool_call_end", payload)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        content = getattr(output, "content", output)
        text = _split_content(content)[0] if isinstance(content, (str, list)) else _to_text(content)
        self._tool_end(run_id, text, None)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._tool_end(run_id, f"Error: {error}", error)


def _describe_output(outputs: Any) -> str:
    if isinstance(outputs, dict) and isinstance(outputs.get("messages"), list) and outputs["messages"]:
        return _split_content(_message_content(outputs["messages"][-1]))[0]
    return _to_text(outputs)


__all__: Sequence[str] = ["AgentFlowCallbackHandler"]
