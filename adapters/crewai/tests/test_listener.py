import collections
import datetime
import json
import os
import sys
from types import SimpleNamespace

import pytest
from crewai import Agent, Crew, Process, Task
from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.tools import tool

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
sys.path.insert(0, os.path.join(HERE, ".."))
from fake_llm import answer, fake_llm, react  # noqa: E402

from agent_flow_crewai import AgentFlowListener  # noqa: E402

# Event types the webview understands (web/hooks/simulation/process-event.ts).
KNOWN_TYPES = {
    "agent_spawn", "agent_complete", "agent_idle", "message", "context_update", "model_detected",
    "tool_call_start", "tool_call_end", "subagent_dispatch", "subagent_return", "permission_requested",
    "graph_structure", "node_start", "node_end",
}


@pytest.fixture
def listener(tmp_path):
    listener = AgentFlowListener(str(tmp_path / "events.jsonl"))
    yield listener
    listener.close()


def events_of(listener):
    listener.flush()
    if not os.path.exists(listener.path):
        return []
    with open(listener.path, encoding="utf-8") as f:
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


@tool("Adder")
def adder(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def agent(role, policy, tools=(), model="demo-model"):
    return Agent(role=role, goal="help", backstory="b", tools=list(tools), llm=fake_llm(policy, model=model),
                 allow_delegation=False)


def test_single_agent_crew(listener):
    mathy = agent("Mathy", react(lambda task: [("Adder", {"a": 2, "b": 3})], lambda task: "5",
                                 thinking=lambda task: "use the adder"), [adder], model="demo-mini")
    crew = Crew(name="calc", agents=[mathy], tasks=[Task(name="sum", description="Add 2 and 3", expected_output="n", agent=mathy)])
    assert str(crew.kickoff()) == "5"
    events = events_of(listener)
    assert all(e["type"] in KNOWN_TYPES for e in events)
    assert events[0]["payload"] == {"name": "calc", "isMain": True, "task": "Add 2 and 3"}
    assert parents(events) == {"calc": None, "Mathy": "calc"}
    assert of_type(events, "model_detected") == [{"agent": "Mathy", "model": "demo-mini"}]
    start = of_type(events, "tool_call_start")
    assert [(p["agent"], p["tool"], p["inputData"]) for p in start] == [("Mathy", "Adder", {"a": 2, "b": 3})]
    assert [p["result"] for p in of_type(events, "tool_call_end")] == ["5"]
    messages = [(p.get("role"), p["agent"]) for p in of_type(events, "message")]
    assert ("thinking", "Mathy") in messages and ("assistant", "Mathy") in messages
    assert hops(events, "calc") == {("__start__", "sum"): 1, ("sum", "__end__"): 1}
    assert events[-1] == {**events[-1], "type": "agent_complete", "payload": {"name": "calc"}}


def test_async_tasks_and_context_form_a_fan_in(listener):
    a = agent("Alpha", answer("A"))
    b = agent("Beta", answer("B"))
    c = agent("Gamma", answer("C"))
    t1 = Task(name="one", description="first", expected_output="x", agent=a, async_execution=True)
    t2 = Task(name="two", description="second", expected_output="x", agent=b, async_execution=True)
    t3 = Task(name="three", description="third", expected_output="x", agent=c, context=[t1, t2])
    Crew(name="parallel", agents=[a, b, c], tasks=[t1, t2, t3]).kickoff()
    events = events_of(listener)
    structure = of_type(events, "graph_structure")[0]
    assert {(e["source"], e["target"]) for e in structure["edges"]} == {
        ("__start__", "one"), ("__start__", "two"), ("one", "three"), ("two", "three"), ("three", "__end__")}
    assert hops(events, "parallel") == {("__start__", "one"): 1, ("__start__", "two"): 1,
                                        ("one", "three"): 1, ("two", "three"): 1, ("three", "__end__"): 1}


def test_hierarchical_delegation_nests_the_coworker(listener):
    worker = agent("Mathy", react(lambda task: [("Adder", {"a": 1, "b": 1})], lambda task: "2"), [adder])
    manager = fake_llm(react(lambda task: [("Delegate work to coworker",
                                            {"task": "add 1 and 1", "context": "math", "coworker": "Mathy"})],
                             lambda task: "done"))
    Crew(name="team", agents=[worker], process=Process.hierarchical, manager_llm=manager,
         tasks=[Task(name="compute", description="compute", expected_output="n")]).kickoff()
    events = events_of(listener)
    tree = parents(events)
    assert tree == {"team": None, "Crew Manager": "team", "Mathy": "Crew Manager"}
    dispatch = next(p for p in of_type(events, "subagent_dispatch") if p["child"] == "Mathy")
    assert dispatch["parent"] == "Crew Manager" and "add 1 and 1" in dispatch["task"]
    tools = [(p["agent"], p["tool"]) for p in of_type(events, "tool_call_start")]
    assert tools == [("Crew Manager", "Delegate work to coworker"), ("Mathy", "Adder")]


class Pipeline(Flow):
    decisions = 0

    @start()
    def begin(self):
        return "go"

    @listen(begin)
    def left(self, _):
        return "L"

    @listen(begin)
    def right(self, _):
        return "R"

    @listen(and_(left, right))
    def merge(self, _):
        writer = agent("Writer", answer("text"))
        crew = Crew(name="inner", agents=[writer],
                    tasks=[Task(name="write", description="write", expected_output="t", agent=writer)])
        return str(crew.kickoff())

    @router(or_(merge, "again"), emit=["redo", "ship"])
    def decide(self, _):
        Pipeline.decisions += 1
        return "redo" if Pipeline.decisions == 1 else "ship"

    @listen("redo")
    def rework(self, _):
        return str(agent("Fixer", answer("fixed")).kickoff("fix it"))

    @listen(rework)
    def again(self, _):
        return "again"

    @listen("ship")
    def done(self, _):
        return "shipped"


def test_flow_structure_routing_and_nesting(listener):
    Pipeline.decisions = 0
    assert Pipeline().kickoff() == "shipped"
    events = events_of(listener)
    structure = next(p for p in of_type(events, "graph_structure") if p["agent"] == "Pipeline")
    edges = {(e["source"], e["target"]): (e["conditional"], e.get("label")) for e in structure["edges"]}
    assert edges[("decide", "rework")] == (True, "redo") and edges[("decide", "done")] == (True, "ship")
    assert edges[("left", "merge")] == (False, None) and edges[("right", "merge")] == (False, None)
    assert hops(events, "Pipeline") == {
        ("__start__", "begin"): 1, ("begin", "left"): 1, ("begin", "right"): 1,
        ("left", "merge"): 1, ("right", "merge"): 1,       # and_ fan-in
        ("merge", "decide"): 1, ("decide", "rework"): 1,   # router label "redo"
        ("rework", "again"): 1, ("again", "decide"): 1,    # the loop
        ("decide", "done"): 1, ("done", "__end__"): 1,
    }
    tree = parents(events)
    assert tree["Pipeline"] is None
    assert tree["inner"] == "Pipeline" and tree["Writer"] == "inner"   # crew kicked off in a method
    assert tree["Fixer"] == "Pipeline"                                 # direct agent.kickoff() in a method
    started = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_start") if p["node"] != "__end__")
    ended = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_end"))
    assert started == ended


def test_events_are_written_in_emission_order(listener):
    t0 = datetime.datetime(2026, 1, 1)

    def event(kind, seq, event_id, parent=None, started=None, **fields):
        return SimpleNamespace(type=kind, timestamp=t0 + datetime.timedelta(milliseconds=seq), emission_sequence=seq,
                               event_id=event_id, parent_event_id=parent, started_event_id=started, **fields)

    worker = SimpleNamespace(role="Solo")
    start = event("agent_execution_started", 1, "a1", agent=worker, task=None, task_prompt="go")
    tool_start = event("tool_usage_started", 2, "t1", parent="a1", tool_name="Adder", tool_args={"a": 1})
    tool_end = event("tool_usage_finished", 3, "t2", parent="a1", started="t1", output="2")
    finish = event("agent_execution_completed", 4, "a2", started="a1", output="done")
    handler = listener._handlers[0][1]
    for e in (finish, tool_end, start, tool_start):  # delivered out of order by the thread pool
        handler(None, e)
    events = events_of(listener)
    assert [e["type"] for e in events] == ["agent_spawn", "message", "tool_call_start", "tool_call_end", "agent_complete"]
    assert of_type(events, "tool_call_end")[0] == {"agent": "Solo", "tool": "Adder", "result": "2"}


def test_close_stops_listening(tmp_path):
    listener = AgentFlowListener(str(tmp_path / "events.jsonl"))
    listener.close()
    mute = agent("Mute", answer("x"))
    Crew(name="quiet", agents=[mute], tasks=[Task(description="t", expected_output="x", agent=mute)]).kickoff()
    listener.close()  # idempotent
    assert not os.path.exists(listener.path)


def test_a_failing_writer_never_breaks_the_crew(listener, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(listener, "_emit", boom)
    mathy = agent("Mathy", react(lambda task: [("Adder", {"a": 2, "b": 2})], lambda task: "4"), [adder])
    assert str(Crew(agents=[mathy], tasks=[Task(description="add", expected_output="n", agent=mathy)]).kickoff()) == "4"
