import json
import os
import sys
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import multi_agent_demo as demo  # noqa: E402

from agent_flow_langgraph import AgentFlowCallbackHandler  # noqa: E402

# Event types the webview understands (web/hooks/simulation/process-event.ts).
KNOWN_TYPES = {
    "agent_spawn", "agent_complete", "agent_idle", "message", "context_update", "model_detected",
    "tool_call_start", "tool_call_end", "subagent_dispatch", "subagent_return", "permission_requested",
    "graph_structure", "node_start", "node_end",
}


@pytest.fixture(autouse=True)
def no_delay(monkeypatch):
    monkeypatch.setattr(demo, "DELAY", 0)


def read_events(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(events):
    out = []
    for e in events:
        p = e["payload"]
        who = p.get("agent") or p.get("name") or f"{p.get('parent')}->{p.get('child')}"
        out.append((e["type"], who, p.get("tool")))
    return out


def run_demo(tmp_path, with_graph=False):
    path = tmp_path / "events.jsonl"
    graph = demo.build_graph()
    handler = AgentFlowCallbackHandler(str(path), graph=graph if with_graph else None)
    graph.invoke(
        {"messages": [HumanMessage("Write a short report on SaaS churn benchmarks")]},
        config={"callbacks": [handler]},
    )
    return handler, read_events(path)


def test_events_match_agent_flow_schema(tmp_path):
    _, events = run_demo(tmp_path, with_graph=True)
    assert events
    times = [e["time"] for e in events]
    assert times == sorted(times)
    for e in events:
        assert set(e) == {"time", "type", "payload"}
        assert e["type"] in KNOWN_TYPES
        assert isinstance(e["time"], (int, float))


def test_supervisor_with_two_subagents(tmp_path):
    _, events = run_demo(tmp_path)
    s = summary(events)

    assert s[0] == ("agent_spawn", "supervisor", None)
    assert events[0]["payload"]["isMain"] is True
    assert s[1] == ("message", "supervisor", None)
    assert events[1]["payload"] == {
        "agent": "supervisor", "role": "user", "content": "Write a short report on SaaS churn benchmarks",
    }

    # Each subagent is dispatched and spawned under the main agent, runs its
    # tools, returns, and completes.
    for child, tools in (("researcher", ["web_search", "read_url"]), ("writer", ["write_file"])):
        dispatch = s.index(("subagent_dispatch", f"supervisor->{child}", None))
        assert s[dispatch + 1] == ("agent_spawn", child, None)
        assert events[dispatch + 1]["payload"]["parent"] == "supervisor"
        ret = s.index(("subagent_return", f"supervisor->{child}", None))
        assert s[ret + 1] == ("agent_complete", child, None)
        between = s[dispatch:ret]
        assert [t for (kind, who, t) in between if kind == "tool_call_start" and who == child] == tools
        assert [t for (kind, who, t) in between if kind == "tool_call_end" and who == child] == tools

    assert s.index(("subagent_return", "supervisor->researcher", None)) < s.index(
        ("subagent_dispatch", "supervisor->writer", None))
    assert s[-1] == ("agent_complete", "supervisor", None)

    # The researcher's final answer is its return summary.
    ret = next(e for e in events if e["type"] == "subagent_return" and e["payload"]["child"] == "researcher")
    assert ret["payload"]["summary"].startswith("Median churn is 3.5%/month")

    # The subagent task is the handoff input (latest human message).
    spawn = next(e for e in events if e["type"] == "agent_spawn" and e["payload"]["name"] == "researcher")
    assert spawn["payload"]["task"] == "Write a short report on SaaS churn benchmarks"


def test_tool_payloads(tmp_path):
    _, events = run_demo(tmp_path)
    start = next(e["payload"] for e in events if e["type"] == "tool_call_start" and e["payload"]["tool"] == "read_url")
    assert start["inputData"] == {"url": "https://example.com/churn-report"}
    assert "churn-report" in start["args"]
    end = next(e["payload"] for e in events if e["type"] == "tool_call_end" and e["payload"]["tool"] == "read_url")
    assert end["agent"] == "researcher"
    assert "median SaaS churn" in end["result"]
    assert "isError" not in end


def test_tool_error_is_reported(tmp_path):
    @tool
    def flaky(x: str) -> str:
        """Always fails."""
        raise ValueError("upstream timeout")

    agent = create_react_agent(
        demo.scripted(
            AIMessage(content="", tool_calls=[demo.call("flaky", "f1", x="1")]),
            AIMessage(content="gave up"),
        ),
        [flaky],
    )
    path = tmp_path / "events.jsonl"
    agent.invoke({"messages": [HumanMessage("go")]}, config={"callbacks": [AgentFlowCallbackHandler(str(path))]})
    end = next(e["payload"] for e in read_events(path) if e["type"] == "tool_call_end")
    assert end["isError"] is True
    assert "upstream timeout" in end["errorMessage"]


def test_nested_subgraph_becomes_grandchild(tmp_path):
    inner = create_react_agent(
        demo.scripted(
            AIMessage(content="", tool_calls=[demo.call("web_search", "n1", query="q")]),
            AIMessage(content="inner done"),
        ),
        [demo.web_search],
    )
    middle = StateGraph(MessagesState)
    middle.add_node("analyst", inner)
    middle.add_edge(START, "analyst")
    middle.add_edge("analyst", END)
    outer = StateGraph(MessagesState)
    outer.add_node("team", middle.compile())
    outer.add_edge(START, "team")
    outer.add_edge("team", END)

    path = tmp_path / "events.jsonl"
    outer.compile(name="boss").invoke(
        {"messages": [HumanMessage("dig")]}, config={"callbacks": [AgentFlowCallbackHandler(str(path))]})
    events = read_events(path)
    parents = {e["payload"]["name"]: e["payload"].get("parent") for e in events if e["type"] == "agent_spawn"}
    assert parents == {"boss": None, "team": "boss", "analyst": "team"}
    tool_start = next(e["payload"] for e in events if e["type"] == "tool_call_start")
    assert tool_start["agent"] == "analyst"


def test_parallel_fan_out_gets_unique_names(tmp_path):
    from langgraph.types import Send

    barrier = threading.Barrier(2, timeout=5)

    @tool
    def lookup(q: str) -> str:
        """Lookup."""
        barrier.wait()  # both workers are in flight at once
        return q

    def make_worker():
        return create_react_agent(
            demo.scripted(
                AIMessage(content="", tool_calls=[demo.call("lookup", "l1", q="x")]),
                AIMessage(content="ok"),
            ),
            [lookup],
        )

    workers = [make_worker(), make_worker()]
    calls = iter(workers)

    def worker(state):
        return next(calls).invoke(state)

    graph = StateGraph(MessagesState)
    graph.add_node("worker", worker)
    graph.add_conditional_edges(START, lambda s: [Send("worker", s), Send("worker", s)])
    graph.add_edge("worker", END)

    path = tmp_path / "events.jsonl"
    graph.compile(name="fanout").invoke(
        {"messages": [HumanMessage("go")]}, config={"callbacks": [AgentFlowCallbackHandler(str(path))]})
    events = read_events(path)
    spawned = sorted(e["payload"]["name"] for e in events if e["type"] == "agent_spawn" and not e["payload"].get("isMain"))
    assert spawned == ["worker", "worker #2"]
    tool_agents = sorted(e["payload"]["agent"] for e in events if e["type"] == "tool_call_start")
    assert tool_agents == ["worker", "worker #2"]


def test_repeat_invocations_share_one_session(tmp_path):
    path = tmp_path / "events.jsonl"
    handler = AgentFlowCallbackHandler(str(path), truncate=True)
    for _ in range(2):
        demo.build_graph().invoke({"messages": [HumanMessage("again")]}, config={"callbacks": [handler]})
    events = read_events(path)
    main_spawns = [e for e in events if e["type"] == "agent_spawn" and e["payload"].get("isMain")]
    assert len(main_spawns) == 2
    assert [e["payload"]["name"] for e in events if e["type"] == "agent_complete"].count("supervisor") == 2
    assert [e["payload"]["child"] for e in events if e["type"] == "subagent_dispatch"] == [
        "researcher", "writer", "researcher", "writer"]


def test_truncate_resets_file(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("stale\n")
    run = AgentFlowCallbackHandler(str(path), truncate=True)
    demo.build_graph().invoke({"messages": [HumanMessage("x")]}, config={"callbacks": [run]})
    assert "stale" not in path.read_text()


def test_async_invoke(tmp_path):
    import asyncio

    path = tmp_path / "events.jsonl"
    asyncio.run(demo.build_graph().ainvoke(
        {"messages": [HumanMessage("async please")]}, config={"callbacks": [AgentFlowCallbackHandler(str(path))]}))
    s = summary(read_events(path))
    assert s[0] == ("agent_spawn", "supervisor", None)
    assert ("tool_call_end", "researcher", "read_url") in s
    assert ("agent_complete", "writer", None) in s
    assert s[-1] == ("agent_complete", "supervisor", None)


# ─── Graph shape ─────────────────────────────────────────────────────────────

def hops(events, agent):
    """(from, to) node transitions recorded for one agent's graph."""
    return [(src, e["payload"]["node"]) for e in events
            if e["type"] == "node_start" and e["payload"]["agent"] == agent
            for src in e["payload"]["from"]]


def test_graph_structure_is_sent_for_main_graph_and_subgraphs(tmp_path):
    _, events = run_demo(tmp_path, with_graph=True)
    structures = {e["payload"]["agent"]: e["payload"] for e in events if e["type"] == "graph_structure"}
    assert set(structures) == {"supervisor", "researcher", "writer"}

    main = structures["supervisor"]
    kinds = {n["id"]: n["kind"] for n in main["nodes"]}
    assert kinds == {"__start__": "start", "supervisor": "node", "researcher": "subgraph",
                     "writer": "subgraph", "__end__": "end"}
    assert {(e["source"], e["target"]) for e in main["edges"]} == {
        ("__start__", "supervisor"), ("supervisor", "researcher"), ("researcher", "writer"), ("writer", "__end__")}

    react = {(e["source"], e["target"], e["conditional"]) for e in structures["researcher"]["edges"]}
    assert ("agent", "tools", True) in react and ("agent", "__end__", True) in react
    assert ("tools", "agent", False) in react

    # Each structure is sent right after its agent spawns.
    for name in structures:
        i = next(i for i, e in enumerate(events) if e["type"] == "agent_spawn" and e["payload"]["name"] == name)
        assert events[i + 1]["type"] == "graph_structure" and events[i + 1]["payload"]["agent"] == name


def test_react_loop_hops_are_recorded(tmp_path):
    _, events = run_demo(tmp_path, with_graph=True)
    assert hops(events, "supervisor") == [
        ("__start__", "supervisor"), ("supervisor", "researcher"), ("researcher", "writer"), ("writer", "__end__")]
    assert hops(events, "researcher") == [
        ("__start__", "agent"), ("agent", "tools"), ("tools", "agent"), ("agent", "tools"),
        ("tools", "agent"), ("agent", "__end__")]
    steps = [e["payload"]["step"] for e in events if e["type"] == "node_start" and e["payload"]["agent"] == "researcher"]
    assert steps == [1, 2, 3, 4, 5, 6]

    # Every node_start except the __end__ markers has a matching node_end.
    starts = [(e["payload"]["agent"], e["payload"]["node"], e["payload"]["step"]) for e in events
              if e["type"] == "node_start" and e["payload"]["node"] != "__end__"]
    ends = [(e["payload"]["agent"], e["payload"]["node"], e["payload"]["step"]) for e in events if e["type"] == "node_end"]
    assert sorted(starts) == sorted(ends)


def test_hops_are_recorded_without_graph_argument(tmp_path):
    _, events = run_demo(tmp_path)
    assert not [e for e in events if e["type"] == "graph_structure"]
    assert hops(events, "researcher")[:3] == [("__start__", "agent"), ("agent", "tools"), ("tools", "agent")]


def build_diamond_loop():
    """a -> (b, c in parallel) -> d (merge) -> a again once, then end."""
    from typing import Annotated, TypedDict
    import operator

    class State(TypedDict):
        log: Annotated[list, operator.add]

    def step(name):
        return lambda state: {"log": [name]}

    g = StateGraph(State)
    for name in "abcd":
        g.add_node(name, step(name))
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge(["b", "c"], "d")
    g.add_conditional_edges("d", lambda s: "a" if s["log"].count("d") < 2 else END, ["a", END])
    return g.compile(name="diamond")


def test_parallel_branches_merge_and_loop(tmp_path):
    graph = build_diamond_loop()
    path = tmp_path / "events.jsonl"
    graph.invoke({"log": []}, config={"callbacks": [AgentFlowCallbackHandler(str(path), graph=graph)]})
    events = read_events(path)
    h = hops(events, "diamond")
    # Fan-out, merge (d reached from both b and c), loop back to a, final exit.
    assert h == [
        ("__start__", "a"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"),
        ("d", "a"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "__end__")]
    structure = next(e["payload"] for e in events if e["type"] == "graph_structure")
    conditional = {(e["source"], e["target"]) for e in structure["edges"] if e["conditional"]}
    assert conditional == {("d", "a"), ("d", "__end__")}


def test_nested_subgraph_structures(tmp_path):
    inner = create_react_agent(
        demo.scripted(AIMessage(content="done")), [demo.web_search])
    middle = StateGraph(MessagesState)
    middle.add_node("analyst", inner)
    middle.add_edge(START, "analyst")
    middle.add_edge("analyst", END)
    outer = StateGraph(MessagesState)
    outer.add_node("team", middle.compile())
    outer.add_edge(START, "team")
    outer.add_edge("team", END)
    graph = outer.compile(name="boss")

    path = tmp_path / "events.jsonl"
    graph.invoke({"messages": [HumanMessage("dig")]}, config={"callbacks": [AgentFlowCallbackHandler(str(path), graph=graph)]})
    events = read_events(path)
    structures = {e["payload"]["agent"]: {n["id"]: n["kind"] for n in e["payload"]["nodes"]}
                  for e in events if e["type"] == "graph_structure"}
    assert structures["boss"]["team"] == "subgraph"
    assert structures["team"]["analyst"] == "subgraph"
    assert set(structures["analyst"]) == {"__start__", "agent", "tools", "__end__"}
    assert hops(events, "team") == [("__start__", "analyst"), ("analyst", "__end__")]


def test_node_error_is_reported(tmp_path):
    def boom(state):
        raise RuntimeError("kaput")

    g = StateGraph(MessagesState)
    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    path = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError):
        g.compile().invoke({"messages": []}, config={"callbacks": [AgentFlowCallbackHandler(str(path))]})
    end = next(e["payload"] for e in read_events(path) if e["type"] == "node_end")
    assert end["node"] == "boom" and "kaput" in end["error"]
