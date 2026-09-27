import collections
import json
import os
import sys
import threading

import pytest
from strands import Agent, tool
from strands.multiagent import GraphBuilder, Swarm

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
sys.path.insert(0, os.path.join(HERE, ".."))
from fake_model import RuleBasedModel, react, tool_results  # noqa: E402

from agent_flow_strands import AgentFlowHooks  # noqa: E402

# Event types the webview understands (web/hooks/simulation/process-event.ts).
KNOWN_TYPES = {
    "agent_spawn", "agent_complete", "agent_idle", "message", "context_update", "model_detected",
    "tool_call_start", "tool_call_end", "subagent_dispatch", "subagent_return", "permission_requested",
    "graph_structure", "node_start", "node_end",
}


@pytest.fixture
def flow(tmp_path):
    return AgentFlowHooks(str(tmp_path / "events.jsonl"))


def events_of(flow):
    flow.flush()
    with open(flow.path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def of_type(events, kind):
    return [e["payload"] for e in events if e["type"] == kind]


def parents(events):
    out = {}
    for p in of_type(events, "agent_spawn"):
        out.setdefault(p["name"], p.get("parent"))
    return out


def hops(events, agent):
    return collections.Counter(
        (src, p["node"]) for p in of_type(events, "node_start") if p["agent"] == agent for src in p["from"])


def say(text):
    return lambda messages, names: {"text": text}


def make(name, policy, tools=(), hooks=(), model_id="demo-model"):
    return Agent(name=name, model=RuleBasedModel(policy, model_id=model_id), tools=list(tools), hooks=list(hooks), callback_handler=None)


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"found {q}"


@tool
def flaky(q: str) -> str:
    """Always fails."""
    raise RuntimeError("upstream exploded")


def test_single_agent_events(flow):
    agent = make("helper", react(lambda task: [[("lookup", {"q": "x"}), ("lookup", {"q": "y"})]],
                                 lambda task, results: "all done", thinking=lambda task: "two lookups"),
                 [lookup], [flow], model_id="demo-sonnet")
    agent("find things")
    events = events_of(flow)
    assert all(e["type"] in KNOWN_TYPES and isinstance(e["time"], (int, float)) for e in events)
    assert events[0] == {**events[0], "type": "agent_spawn", "payload": {"name": "helper", "isMain": True, "task": "find things"}}
    assert of_type(events, "message")[0] == {"agent": "helper", "role": "user", "content": "find things"}
    assert of_type(events, "model_detected") == [{"agent": "helper", "model": "demo-sonnet"}]
    roles = [p.get("role") for p in of_type(events, "message")]
    assert roles == ["user", "thinking", "assistant"]
    starts = of_type(events, "tool_call_start")
    assert [(p["tool"], p["inputData"]) for p in starts] == [("lookup", {"q": "x"}), ("lookup", {"q": "y"})]
    ends = of_type(events, "tool_call_end")
    assert sorted(p["result"] for p in ends) == ["found x", "found y"]
    assert all(p["tokens"] > 0 for p in of_type(events, "context_update"))
    assert events[-1] == {**events[-1], "type": "agent_complete", "payload": {"name": "helper"}}


def test_tool_errors_are_reported(flow):
    agent = make("helper", react(lambda task: [[("flaky", {"q": "x"})]], lambda task, results: "gave up"), [flaky], [flow])
    agent("go")
    ends = of_type(events_of(flow), "tool_call_end")
    assert ends and all(p["isError"] for p in ends)
    assert "upstream exploded" in ends[0]["errorMessage"]


def test_agents_as_tools_become_subagents_even_in_parallel(flow):
    barrier = threading.Barrier(3, timeout=10)

    @tool
    def delegate(topic: str) -> str:
        """Hand a topic to a fresh specialist agent."""
        barrier.wait()  # all three specialists are in flight together
        # A slow model keeps all three specialist runs in flight at once
        specialist = Agent(name="specialist", model=RuleBasedModel(say(f"notes on {topic}"), delay=lambda: 0.2),
                           hooks=[flow], callback_handler=None)
        return str(specialist(f"research {topic}"))

    boss = make("boss", react(lambda task: [[("delegate", {"topic": t}) for t in ("a", "b", "c")]],
                              lambda task, results: "merged"), [delegate], [flow])
    boss("plan")
    events = events_of(flow)
    tree = parents(events)
    assert tree["boss"] is None
    specialists = sorted(n for n, p in tree.items() if p == "boss")
    assert specialists == ["specialist", "specialist #2", "specialist #3"]  # concurrent: unique names
    dispatch = of_type(events, "subagent_dispatch")
    assert sorted(p["task"] for p in dispatch) == ["research a", "research b", "research c"]
    returns = of_type(events, "subagent_return")
    assert sorted(p["summary"] for p in returns) == ["notes on a", "notes on b", "notes on c"]


def test_sequential_agents_reuse_their_name(flow):
    @tool
    def delegate(topic: str) -> str:
        """Hand a topic to a fresh helper agent."""
        return str(make("helper", say("ok"), hooks=[flow])(topic))

    boss = make("boss", react(lambda task: [[("delegate", {"topic": "a"})], [("delegate", {"topic": "b"})]],
                              lambda task, results: "done"), [delegate], [flow])
    boss("go")
    names = [p["name"] for p in of_type(events_of(flow), "agent_spawn")]
    assert names == ["boss", "helper", "helper"]


def build_graph(flow, reviewer_policy=None):
    reviewer_policy = reviewer_policy or say("APPROVE")
    gb = GraphBuilder()
    gb.add_node(make("planner_agent", say("plan")), "plan")
    gb.add_node(make("left", say("L")), "left")
    gb.add_node(make("right", say("R")), "right")
    gb.add_node(make("merge", say("merged")), "merge")
    gb.add_node(make("review", reviewer_policy), "review")
    gb.add_node(make("ship", say("shipped")), "ship")
    gb.add_edge("plan", "left")
    gb.add_edge("plan", "right")
    gb.add_edge("left", "merge")
    gb.add_edge("right", "merge")
    gb.add_edge("merge", "review")
    gb.add_edge("review", "merge", condition=lambda state: "REVISE" in str(state.results["review"].result))
    gb.add_edge("review", "ship", condition=lambda state: "APPROVE" in str(state.results["review"].result))
    gb.reset_on_revisit(True)
    gb.set_max_node_executions(20)
    gb.set_graph_id("pipeline")
    return flow.instrument(gb.build())


def test_graph_structure(flow):
    build_graph(flow)("do it")
    structure = of_type(events_of(flow), "graph_structure")[0]
    assert structure["agent"] == "pipeline"
    assert [n["id"] for n in structure["nodes"]] == ["__start__", "plan", "left", "right", "merge", "review", "ship", "__end__"]
    edges = {(e["source"], e["target"]): e["conditional"] for e in structure["edges"]}
    assert edges == {
        ("__start__", "plan"): False, ("plan", "left"): False, ("plan", "right"): False,
        ("left", "merge"): False, ("right", "merge"): False, ("merge", "review"): False,
        ("review", "merge"): True, ("review", "ship"): True, ("ship", "__end__"): False,
    }


def test_graph_parallel_merge_and_conditional_loop(flow):
    verdicts = iter(["REVISE", "APPROVE"])
    graph = build_graph(flow, lambda messages, names: {"text": next(verdicts)})
    result = graph("do it")
    assert result.status.value == "completed"
    events = events_of(flow)
    h = hops(events, "pipeline")
    assert h == {
        ("__start__", "plan"): 1, ("plan", "left"): 1, ("plan", "right"): 1,
        ("left", "merge"): 1, ("right", "merge"): 1,       # merge reached from both branches
        ("merge", "review"): 2, ("review", "merge"): 1,     # the loop, taken once
        ("review", "ship"): 1, ("ship", "__end__"): 1,
    }
    # Node agents are the graph's subagents, named after their node
    tree = parents(events)
    assert {n for n, p in tree.items() if p == "pipeline"} == {"plan", "left", "right", "merge", "review", "ship"}
    # left and right ran in the same batch
    steps = {(p["node"], p["step"]) for p in of_type(events, "node_start") if p["agent"] == "pipeline"}
    assert ("left", 2) in steps and ("right", 2) in steps and ("merge", 3) in steps
    started = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_start") if p["node"] != "__end__")
    ended = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_end"))
    assert started == ended


def test_nested_graph_is_a_subgraph(flow):
    inner = GraphBuilder()
    inner.add_node(make("a", say("A")), "a")
    inner.add_node(make("b", say("B")), "b")
    inner.add_edge("a", "b")
    inner.set_graph_id("inner")
    outer = GraphBuilder()
    outer.add_node(make("start", say("go")), "kickoff")
    outer.add_node(inner.build(), "team")
    outer.add_edge("kickoff", "team")
    outer.set_graph_id("outer")
    flow.instrument(outer.build())("run")
    events = events_of(flow)
    structures = {p["agent"]: {n["id"]: n["kind"] for n in p["nodes"]} for p in of_type(events, "graph_structure")}
    assert structures["outer"]["team"] == "subgraph"
    assert set(structures["team"]) == {"__start__", "a", "b", "__end__"}
    tree = parents(events)
    assert tree["team"] == "outer" and tree["a"] == "team" and tree["b"] == "team"
    assert hops(events, "team") == {("__start__", "a"): 1, ("a", "b"): 1, ("b", "__end__"): 1}


def test_swarm_handoffs_are_hops(flow):
    def hand_to(target):
        def policy(messages, names):
            if target is None or tool_results(messages, ["handoff_to_agent"]):
                return {"text": "done"}
            return {"tools": [("handoff_to_agent", {"agent_name": target, "message": "over to you"})]}
        return policy

    swarm = Swarm([make("triage", hand_to("coder")), make("coder", hand_to("tester")), make("tester", hand_to(None))])
    swarm.id = "support"
    flow.instrument(swarm)("fix the bug")
    events = events_of(flow)
    structure = of_type(events, "graph_structure")[0]
    assert [n["id"] for n in structure["nodes"]] == ["__start__", "triage", "coder", "tester", "__end__"]
    assert structure["edges"] == [{"source": "__start__", "target": "triage", "conditional": False}]
    assert hops(events, "support") == {
        ("__start__", "triage"): 1, ("triage", "coder"): 1, ("coder", "tester"): 1, ("tester", "__end__"): 1}
    handoffs = [p for p in of_type(events, "tool_call_start") if p["tool"] == "handoff_to_agent"]
    assert [(p["agent"], p["inputData"]["agent_name"]) for p in handoffs] == [("triage", "coder"), ("coder", "tester")]


def test_instrument_is_idempotent(flow):
    agent = make("solo", say("hi"))
    flow.instrument(agent)
    flow.instrument(agent)
    agent("hello")
    assert [p["name"] for p in of_type(events_of(flow), "agent_spawn")] == ["solo"]


def test_a_failing_writer_never_breaks_the_agent(flow, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(flow, "_emit", boom)
    agent = make("sturdy", react(lambda task: [[("lookup", {"q": "x"})]], lambda task, results: "still fine"), [lookup], [flow])
    assert str(agent("go")).strip() == "still fine"
