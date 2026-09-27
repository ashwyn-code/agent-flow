import asyncio
import collections
import json
import os
import sys

import pytest
from agent_framework import Agent, AgentExecutor, AgentExecutorResponse, WorkflowBuilder, WorkflowContext, WorkflowExecutor, executor, tool

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
sys.path.insert(0, os.path.join(HERE, ".."))
from fake_client import RuleBasedChatClient, react  # noqa: E402

from agent_flow_agent_framework import AgentFlow  # noqa: E402

# Event types the webview understands (web/hooks/simulation/process-event.ts).
KNOWN_TYPES = {
    "agent_spawn", "agent_complete", "agent_idle", "message", "context_update", "model_detected",
    "tool_call_start", "tool_call_end", "subagent_dispatch", "subagent_return", "permission_requested",
    "graph_structure", "node_start", "node_end",
}


@pytest.fixture
def flow(tmp_path):
    return AgentFlow(str(tmp_path / "events.jsonl"))


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
    return lambda messages: {"text": text}


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"found {q}"


@tool
def flaky(q: str) -> str:
    """Always fails."""
    raise RuntimeError("upstream exploded")


def helper(flow, model="demo-gpt"):
    policy = react(lambda task: [[("lookup", {"q": "x"}), ("lookup", {"q": "y"})]],
                   lambda task, results: "all done", thinking=lambda task: "two lookups")
    return Agent(RuleBasedChatClient(policy, model=model), name="helper", tools=[lookup], middleware=flow.middleware)


@pytest.mark.parametrize("stream", [False, True])
def test_single_agent_events(flow, stream):
    agent = helper(flow)

    async def go():
        if not stream:
            return await agent.run("find things")
        s = agent.run("find things", stream=True)
        async for _ in s:
            pass
        return await s.get_final_response()

    assert asyncio.run(go()).text == "all done"
    events = events_of(flow)
    assert all(e["type"] in KNOWN_TYPES for e in events)
    assert events[0]["payload"] == {"name": "helper", "isMain": True, "task": "find things"}
    assert [p.get("role") for p in of_type(events, "message")] == ["user", "thinking", "assistant"]
    assert of_type(events, "model_detected") == [{"agent": "helper", "model": "demo-gpt"}]
    starts = of_type(events, "tool_call_start")
    assert sorted((p["agent"], p["tool"], p["inputData"]["q"]) for p in starts) == [("helper", "lookup", "x"), ("helper", "lookup", "y")]
    assert sorted(p["result"] for p in of_type(events, "tool_call_end")) == ["found x", "found y"]
    assert of_type(events, "context_update") and all(p["tokens"] > 0 for p in of_type(events, "context_update"))
    assert events[-1] == {**events[-1], "type": "agent_complete", "payload": {"name": "helper"}}


def test_tool_errors_are_reported(flow):
    agent = Agent(RuleBasedChatClient(react(lambda task: [[("flaky", {"q": "x"})]], lambda task, results: "gave up")),
                  name="helper", tools=[flaky], middleware=flow.middleware)
    asyncio.run(agent.run("go"))
    ends = of_type(events_of(flow), "tool_call_end")
    assert len(ends) == 2 and all(p["isError"] for p in ends)  # tried, retried once
    assert "upstream exploded" in ends[0]["errorMessage"]


def test_agents_as_tools_become_subagents_even_concurrently(flow):
    researcher = Agent(RuleBasedChatClient(react(lambda task: [[("lookup", {"q": task})]],
                                                  lambda task, results: f"notes on {task}")), name="researcher", tools=[lookup])
    boss = Agent(RuleBasedChatClient(react(lambda task: [[("researcher", {"task": t}) for t in ("a", "b", "c")]],
                                            lambda task, results: "merged")),
                 name="boss", tools=[researcher.as_tool()])
    flow.instrument(boss)  # also finds and instruments the as_tool() agent
    asyncio.run(boss.run("plan"))
    events = events_of(flow)
    tree = parents(events)
    assert tree["boss"] is None
    # One agent object, three concurrent runs: three distinct subagents
    assert sorted(n for n, p in tree.items() if p == "boss") == ["researcher", "researcher #2", "researcher #3"]
    assert sorted(p["task"] for p in of_type(events, "subagent_dispatch")) == ["a", "b", "c"]
    assert sorted(p["summary"] for p in of_type(events, "subagent_return")) == ["notes on a", "notes on b", "notes on c"]
    # The subagents' own tool calls are attributed to them, not to boss
    inner = [p["agent"] for p in of_type(events, "tool_call_start") if p["tool"] == "lookup"]
    assert sorted(inner) == ["researcher", "researcher #2", "researcher #3"]


def test_instrument_is_idempotent(flow):
    agent = Agent(RuleBasedChatClient(say("hi")), name="solo")
    flow.instrument(agent)
    flow.instrument(agent)
    asyncio.run(agent.run("hello"))
    assert [p["name"] for p in of_type(events_of(flow), "agent_spawn")] == ["solo"]


def build_workflow(flow):
    @executor(id="split")
    async def split(msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(msg)

    @executor(id="join")
    async def join(msgs: list[AgentExecutorResponse], ctx: WorkflowContext[str, str]) -> None:
        await ctx.yield_output(f"{len(msgs)} results")

    @executor(id="small")
    async def small(msg: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.yield_output("small job")

    left = AgentExecutor(Agent(RuleBasedChatClient(say("L")), name="left"), id="left")
    right = AgentExecutor(Agent(RuleBasedChatClient(say("R")), name="right"), id="right")

    @executor(id="route")
    async def route(msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(msg)

    workflow = (WorkflowBuilder(start_executor=route, name="pipeline", output_from=[join, small])
                .add_edge(route, split, condition=lambda m: "big" in m)
                .add_edge(route, small, condition=lambda m: "big" not in m)
                .add_fan_out_edges(split, [left, right])
                .add_fan_in_edges([left, right], join)
                .build())
    return flow.instrument(workflow)


def test_workflow_structure(flow):
    asyncio.run(build_workflow(flow).run("big job"))
    structure = of_type(events_of(flow), "graph_structure")[0]
    assert structure["agent"] == "pipeline"
    assert {n["id"] for n in structure["nodes"]} == {"__start__", "route", "split", "small", "left", "right", "join", "__end__"}
    edges = {(e["source"], e["target"]): e["conditional"] for e in structure["edges"]}
    assert edges == {
        ("__start__", "route"): False, ("route", "split"): True, ("route", "small"): True,
        ("split", "left"): False, ("split", "right"): False, ("left", "join"): False, ("right", "join"): False,
        ("small", "__end__"): False, ("join", "__end__"): False,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_workflow_routing_fan_out_and_fan_in(flow, stream):
    workflow = build_workflow(flow)

    async def go():
        if not stream:
            return (await workflow.run("big job")).get_outputs()
        outputs = []
        async for event in workflow.run("big job", stream=True):
            if event.type == "output":
                outputs.append(event.data)
        return outputs

    assert asyncio.run(go()) == ["2 results"]
    events = events_of(flow)
    assert hops(events, "pipeline") == {
        ("__start__", "route"): 1, ("route", "split"): 1,              # the untaken branch has no hop
        ("split", "left"): 1, ("split", "right"): 1,
        ("left", "join"): 1, ("right", "join"): 1,                   # fan-in from both
        ("join", "__end__"): 1,
    }
    join = next(p for p in of_type(events, "node_start") if p["node"] == "join")
    assert join["from"] == ["left", "right"] and join["step"] == 4
    tree = parents(events)
    assert tree["pipeline"] is None and tree["left"] == "pipeline" and tree["right"] == "pipeline"
    assert events[-1] == {**events[-1], "type": "agent_complete", "payload": {"name": "pipeline"}}


def test_nested_workflow_is_a_subgraph(flow):
    @executor(id="inner_a")
    async def inner_a(msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(msg + "!")

    @executor(id="inner_b")
    async def inner_b(msg: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.yield_output(msg + "?")

    inner = WorkflowBuilder(start_executor=inner_a, name="inner").add_edge(inner_a, inner_b).build()

    @executor(id="kickoff")
    async def kickoff(msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(msg)

    @executor(id="finish")
    async def finish(msg: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.yield_output("done: " + msg)

    team = WorkflowExecutor(inner, id="team")
    outer = WorkflowBuilder(start_executor=kickoff, name="outer").add_edge(kickoff, team).add_edge(team, finish).build()
    flow.instrument(outer)
    assert asyncio.run(outer.run("go")).get_outputs() == ["done: go!?"]
    events = events_of(flow)
    structures = {p["agent"]: {n["id"]: n["kind"] for n in p["nodes"]} for p in of_type(events, "graph_structure")}
    assert structures["outer"]["team"] == "subgraph"
    assert set(structures["team"]) == {"__start__", "inner_a", "inner_b", "__end__"}
    assert parents(events)["team"] == "outer"
    assert hops(events, "team") == {("__start__", "inner_a"): 1, ("inner_a", "inner_b"): 1, ("inner_b", "__end__"): 1}
    assert hops(events, "outer")[("team", "finish")] == 1


def test_failing_executor_is_marked_and_run_completes(flow):
    @executor(id="boom")
    async def boom(msg: str, ctx: WorkflowContext[str]) -> None:
        raise ValueError("kaput")

    workflow = flow.instrument(WorkflowBuilder(start_executor=boom, name="fragile").build())

    async def go():
        try:
            await workflow.run("x")
        except Exception:
            pass

    asyncio.run(go())
    events = events_of(flow)
    end = next(p for p in of_type(events, "node_end") if p["node"] == "boom")
    assert "kaput" in end["error"]
    assert events[-1]["type"] == "agent_complete" and events[-1]["payload"] == {"name": "fragile"}


def test_a_failing_writer_never_breaks_the_run(flow, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(flow, "_emit", boom)
    assert asyncio.run(helper(flow).run("go")).text == "all done"
    assert asyncio.run(build_workflow(flow).run("big job")).get_outputs() == ["2 results"]
