"""End-to-end test: the trip planner demo through the plugin."""
import asyncio
import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import trip_planner_demo as demo  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("adk") / "events.jsonl"
    # A small delay keeps the parallel agents and tool calls in flight together.
    final = asyncio.run(demo.run(str(path), delay=0.02))
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    return final, events


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


def test_run_completes(run):
    final, events = run
    assert final == "Booked: flight and hotel confirmed."
    assert events[-1]["type"] == "agent_complete" and events[-1]["payload"] == {"name": "trip_planner"}


def test_agent_tree(run):
    _, events = run
    assert parents(events) == {
        "trip_planner": None,
        "intake": "trip_planner", "research": "trip_planner", "itinerary_loop": "trip_planner", "booking": "trip_planner",
        "flights": "research", "hotels": "research", "weather": "research",
        "planner": "itinerary_loop", "critic": "itinerary_loop",
        "budget_analyst": "planner",        # AgentTool
        "reservations": "booking",          # transfer_to_agent
    }


def test_workflow_graphs(run):
    _, events = run
    assert hops(events, "trip_planner") == {
        ("__start__", "intake"): 1, ("intake", "research"): 1, ("research", "itinerary_loop"): 1,
        ("itinerary_loop", "booking"): 1, ("booking", "__end__"): 1}
    assert hops(events, "research") == {(s, t): 1 for s, t in [
        ("__start__", "flights"), ("__start__", "hotels"), ("__start__", "weather"),
        ("flights", "__end__"), ("hotels", "__end__"), ("weather", "__end__")]}
    assert hops(events, "itinerary_loop") == {("__start__", "planner"): 1, ("planner", "critic"): 2,
                                              ("critic", "planner"): 1, ("critic", "__end__"): 1}
    kinds = {n["id"]: n["kind"] for p in of_type(events, "graph_structure") if p["agent"] == "trip_planner" for n in p["nodes"]}
    assert kinds["research"] == "subgraph" and kinds["itinerary_loop"] == "subgraph" and kinds["intake"] == "node"


def test_parallel_research_overlaps(run):
    _, events = run
    order = [(e["type"], e["payload"]["node"]) for e in events
             if e["type"] in ("node_start", "node_end") and e["payload"]["agent"] == "research"]
    starts = [order.index(("node_start", n)) for n in ("flights", "hotels", "weather")]
    first_end = min(i for i, (kind, _) in enumerate(order) if kind == "node_end")
    assert max(starts) < first_end


def test_tools_failure_and_transfer(run):
    _, events = run
    flights = [bool(p.get("isError")) for p in of_type(events, "tool_call_end") if p["tool"] == "search_flights"]
    assert flights == [True, False]
    tools = [(p["agent"], p["tool"]) for p in of_type(events, "tool_call_start")]
    assert ("critic", "exit_loop") in tools and ("booking", "transfer_to_agent") in tools
    assert [t for a, t in tools if a == "hotels"] == ["search_hotels", "search_hotels"]


def test_everything_finishes(run):
    _, events = run
    started = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_start") if p["node"] != "__end__")
    ended = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_end"))
    assert started == ended
    assert collections.Counter(p["name"] for p in of_type(events, "agent_spawn")) == \
        collections.Counter(p["name"] for p in of_type(events, "agent_complete"))
    assert len(of_type(events, "tool_call_start")) == len(of_type(events, "tool_call_end"))


def test_models_and_reasoning(run):
    _, events = run
    models = {p["agent"]: p["model"] for p in of_type(events, "model_detected")}
    assert models["planner"] == "demo-gemini-2.5-pro" and models["weather"] == "demo-gemini-2.5-flash"
    roles = collections.Counter(p.get("role") for p in of_type(events, "message"))
    assert roles["thinking"] >= 4 and roles["user"] == 1
