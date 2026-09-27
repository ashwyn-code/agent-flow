import asyncio
import collections
import json
import os
import sys

import pytest
from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import exit_loop
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
sys.path.insert(0, os.path.join(HERE, ".."))
from fake_llm import fake_llm, react  # noqa: E402

from agent_flow_adk import AgentFlowPlugin  # noqa: E402

# Event types the webview understands (web/hooks/simulation/process-event.ts).
KNOWN_TYPES = {
    "agent_spawn", "agent_complete", "agent_idle", "message", "context_update", "model_detected",
    "tool_call_start", "tool_call_end", "subagent_dispatch", "subagent_return", "permission_requested",
    "graph_structure", "node_start", "node_end",
}


@pytest.fixture
def plugin(tmp_path):
    return AgentFlowPlugin(str(tmp_path / "events.jsonl"))


def run(plugin, root, text="go", streaming=False):
    async def go():
        app = App(name="test_app", root_agent=root, plugins=[plugin])
        runner = Runner(app=app, session_service=InMemorySessionService())
        session = await runner.session_service.create_session(app_name="test_app", user_id="u")
        config = RunConfig(streaming_mode=StreamingMode.SSE) if streaming else None
        kwargs = {"run_config": config} if config else {}
        async for _ in runner.run_async(user_id="u", session_id=session.id,
                                        new_message=types.Content(role="user", parts=[types.Part(text=text)]), **kwargs):
            pass
    asyncio.run(go())


def events_of(plugin):
    if not os.path.exists(plugin.path):
        return []
    with open(plugin.path, encoding="utf-8") as f:
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
    return lambda contents: {"text": text}


async def lookup(q: str) -> dict:
    """Look something up."""
    return {"status": "ok", "value": f"found {q}"}


async def broken(q: str) -> dict:
    """Reports an error."""
    return {"status": "error", "error_message": "upstream down"}


async def explodes(q: str) -> dict:
    """Raises."""
    raise ValueError("kaboom")


def helper(name="helper", model="demo-model"):
    return LlmAgent(name=name, tools=[lookup], model=fake_llm(react(
        lambda task: [[("lookup", {"q": "x"}), ("lookup", {"q": "y"})]], lambda task, r: "all done",
        thinking=lambda task: "two lookups"), model=model))


@pytest.mark.parametrize("streaming", [False, True])
def test_single_agent_events(plugin, streaming):
    run(plugin, helper(), "find things", streaming=streaming)
    events = events_of(plugin)
    assert all(e["type"] in KNOWN_TYPES for e in events)
    assert events[0]["payload"] == {"name": "helper", "isMain": True, "task": "find things"}
    assert of_type(events, "message")[0] == {"agent": "helper", "role": "user", "content": "find things"}
    assert of_type(events, "model_detected") == [{"agent": "helper", "model": "demo-model"}]
    starts = of_type(events, "tool_call_start")
    assert sorted((p["tool"], p["inputData"]["q"]) for p in starts) == [("lookup", "x"), ("lookup", "y")]
    assert [p.get("role") for p in of_type(events, "message")] == ["user", "thinking", "assistant"]
    assert all(p["tokens"] > 0 for p in of_type(events, "context_update"))
    assert events[-1] == {**events[-1], "type": "agent_complete", "payload": {"name": "helper"}}


def test_transfer_to_sub_agent(plugin):
    billing = LlmAgent(name="billing", description="refunds", model=fake_llm(say("refunded")))
    coordinator = LlmAgent(name="coordinator", sub_agents=[billing], model=fake_llm(react(
        lambda task: [[("transfer_to_agent", {"agent_name": "billing"})]], lambda task, r: "routed")))
    run(plugin, coordinator, "refund please")
    events = events_of(plugin)
    assert parents(events) == {"coordinator": None, "billing": "coordinator"}
    assert [(p["agent"], p["tool"]) for p in of_type(events, "tool_call_start")] == [("coordinator", "transfer_to_agent")]
    # ADK never fires after_agent for the coordinator; it is closed with the run
    completed = [p["name"] for p in of_type(events, "agent_complete")]
    assert completed == ["billing", "coordinator"]


def test_agent_tools_become_subagents_even_concurrently(plugin):
    researcher = LlmAgent(name="researcher", tools=[lookup], model=fake_llm(react(
        lambda task: [[("lookup", {"q": task})]], lambda task, r: f"notes on {task}")))
    boss = LlmAgent(name="boss", tools=[AgentTool(agent=researcher)], model=fake_llm(react(
        lambda task: [[("researcher", {"request": t}) for t in ("a", "b", "c")]], lambda task, r: "merged")))
    run(plugin, boss, "plan")
    events = events_of(plugin)
    tree = parents(events)
    assert sorted(n for n, p in tree.items() if p == "boss") == ["researcher", "researcher #2", "researcher #3"]
    assert sorted(p["summary"] for p in of_type(events, "subagent_return")) == ["notes on a", "notes on b", "notes on c"]
    inner = sorted(p["agent"] for p in of_type(events, "tool_call_start") if p["tool"] == "lookup")
    assert inner == ["researcher", "researcher #2", "researcher #3"]


def test_sequential_and_parallel_agents_are_graphs(plugin):
    a, b, c = helper("a"), helper("b"), LlmAgent(name="c", model=fake_llm(say("merged")))
    pipeline = SequentialAgent(name="pipeline", sub_agents=[ParallelAgent(name="fanout", sub_agents=[a, b]), c])
    run(plugin, pipeline)
    events = events_of(plugin)
    structures = {p["agent"]: p for p in of_type(events, "graph_structure")}
    assert {n["id"]: n["kind"] for n in structures["pipeline"]["nodes"]}["fanout"] == "subgraph"
    assert parents(events) == {"pipeline": None, "fanout": "pipeline", "a": "fanout", "b": "fanout", "c": "pipeline"}
    assert hops(events, "pipeline") == {("__start__", "fanout"): 1, ("fanout", "c"): 1, ("c", "__end__"): 1}
    assert hops(events, "fanout") == {("__start__", "a"): 1, ("__start__", "b"): 1, ("a", "__end__"): 1, ("b", "__end__"): 1}
    # a and b overlap: both start before either finishes
    order = [(e["type"], e["payload"]["node"]) for e in events if e["type"] in ("node_start", "node_end") and e["payload"]["agent"] == "fanout"]
    assert order.index(("node_start", "b")) < order.index(("node_end", "a"))


def test_loop_agent_draws_its_iterations(plugin):
    calls = []

    def checker_policy(contents):
        calls.append(1)
        return {"text": "again"} if len(calls) == 1 else {"tools": [("exit_loop", {})]}

    loop = LoopAgent(name="refine", max_iterations=5, sub_agents=[
        LlmAgent(name="worker", model=fake_llm(say("draft"))),
        LlmAgent(name="checker", tools=[exit_loop], model=fake_llm(checker_policy)),
    ])
    run(plugin, loop)
    events = events_of(plugin)
    structure = of_type(events, "graph_structure")[0]
    loop_edges = {(e["source"], e["target"]) for e in structure["edges"] if e["conditional"]}
    assert ("checker", "worker") in loop_edges and ("checker", "__end__") in loop_edges
    assert hops(events, "refine") == {("__start__", "worker"): 1, ("worker", "checker"): 2,
                                      ("checker", "worker"): 1, ("checker", "__end__"): 1}


def test_tool_errors(plugin):
    agent = LlmAgent(name="worker", tools=[broken], model=fake_llm(react(
        lambda task: [[("broken", {"q": "x"})]], lambda task, r: "gave up")))
    run(plugin, agent)
    ends = of_type(events_of(plugin), "tool_call_end")
    assert len(ends) == 2 and all(p["isError"] for p in ends)   # tried, retried once
    assert ends[0]["errorMessage"] == "upstream down"


def test_raising_tool_ends_the_run_cleanly(plugin):
    agent = LlmAgent(name="worker", tools=[explodes], model=fake_llm(react(
        lambda task: [[("explodes", {"q": "x"})]], lambda task, r: "never")))
    with pytest.raises(ValueError):
        run(plugin, agent)
    events = events_of(plugin)
    end = of_type(events, "tool_call_end")[0]
    assert end["isError"] and "kaboom" in end["errorMessage"]
    assert events[-1]["type"] == "agent_complete" and events[-1]["payload"] == {"name": "worker"}


def test_callbacks_never_interfere(plugin, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(plugin, "_emit", boom)
    run(plugin, helper())   # completes; ADK would abort the run if a plugin raised
