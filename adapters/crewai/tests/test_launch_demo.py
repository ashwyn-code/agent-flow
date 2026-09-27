"""End-to-end test: the launch Flow demo through the listener."""
import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import launch_flow_demo as demo  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("crewai") / "events.jsonl"
    # A small delay keeps the parallel branches and async tasks in flight together.
    result = demo.run(str(path), delay=0.02)
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    return result, events


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
    result, events = run
    assert result == "Live at /blog/introducing-pulse."
    assert events[-1]["type"] == "agent_complete" and events[-1]["payload"] == {"name": "LaunchFlow"}


def test_agent_tree(run):
    _, events = run
    tree = parents(events)
    assert len(tree) == 14
    assert tree["LaunchFlow"] is None
    assert {n for n, p in tree.items() if p == "LaunchFlow"} == {
        "research_crew", "creative_crew", "copy_crew", "Line Editor", "Publisher"}
    assert {n for n, p in tree.items() if p == "research_crew"} == {"Market Analyst", "Competitor Analyst", "Insights Lead"}
    assert {n for n, p in tree.items() if p == "creative_crew"} == {"Designer", "Brand Reviewer"}
    assert tree["Crew Manager"] == "copy_crew"
    assert tree["Copywriter"] == "Crew Manager" and tree["Editor"] == "Crew Manager"   # delegation

    def depth(name):
        return 0 if tree[name] is None else 1 + depth(tree[name])
    assert max(depth(n) for n in tree) == 3


def test_flow_graph(run):
    _, events = run
    structure = next(p for p in of_type(events, "graph_structure") if p["agent"] == "LaunchFlow")
    routed = {(e["source"], e["target"], e.get("label")) for e in structure["edges"] if e["conditional"]}
    assert routed == {("review", "polish", "revise"), ("review", "publish", "approve")}
    h = hops(events, "LaunchFlow")
    assert h[("brief", "research")] == 1 and h[("brief", "creative")] == 1
    assert h[("research", "draft")] == 1 and h[("creative", "draft")] == 1        # and_ fan-in
    assert h[("review", "polish")] == 1 and h[("polished", "review")] == 1       # revise loop, once
    assert h[("review", "publish")] == 1 and h[("publish", "__end__")] == 1


def test_parallel_branches_overlap(run):
    _, events = run
    order = [(e["type"], e["payload"].get("child")) for e in events if e["type"] in ("subagent_dispatch", "subagent_return")]

    def overlapping(names):
        started = [i for i, (kind, child) in enumerate(order) if kind == "subagent_dispatch" and child in names]
        first_done = min(i for i, (kind, child) in enumerate(order) if kind == "subagent_return" and child in names)
        return len(started) >= len(names) and sorted(started)[len(names) - 1] < first_done

    assert overlapping(["Market Analyst", "Competitor Analyst"])    # async tasks run concurrently

    # Flow fan-out: both methods start before either finishes. (CrewAI then
    # runs the two crew kickoffs one after the other, which the trace shows.)
    flow = [(e["type"], e["payload"]["node"]) for e in events
            if e["type"] in ("node_start", "node_end") and e["payload"]["agent"] == "LaunchFlow"]
    starts = [flow.index(("node_start", m)) for m in ("research", "creative")]
    first_end = min(flow.index(("node_end", m)) for m in ("research", "creative"))
    assert max(starts) < first_end


def test_flow_methods_link_to_their_crews(run):
    _, events = run
    links = {n["id"]: n["child"] for p in of_type(events, "graph_structure") if p["agent"] == "LaunchFlow"
             for n in p["nodes"] if n.get("child")}
    assert links == {"research": "research_crew", "creative": "creative_crew", "draft": "copy_crew"}


def test_crew_graphs(run):
    _, events = run
    assert hops(events, "research_crew") == {
        ("__start__", "market_scan"): 1, ("__start__", "competitor_scan"): 1,
        ("market_scan", "insights"): 1, ("competitor_scan", "insights"): 1, ("insights", "__end__"): 1}
    assert hops(events, "creative_crew") == {
        ("__start__", "design_assets"): 1, ("design_assets", "brand_check"): 1, ("brand_check", "__end__"): 1}


def test_delegation_and_tool_failure(run):
    _, events = run
    manager_tools = [p["tool"] for p in of_type(events, "tool_call_start") if p["agent"] == "Crew Manager"]
    assert manager_tools == ["Delegate work to coworker", "Ask question to coworker"]
    pricing = [bool(p.get("isError")) for p in of_type(events, "tool_call_end")
               if p["agent"] == "Competitor Analyst" and p["tool"] == "Pricing Lookup"]
    assert pricing[0] is True and pricing[-1] is False   # failed, then the retry worked


def test_everything_finishes(run):
    _, events = run
    started = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_start") if p["node"] != "__end__")
    ended = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_end"))
    assert started == ended
    assert collections.Counter(p["name"] for p in of_type(events, "agent_spawn")) == \
        collections.Counter(p["name"] for p in of_type(events, "agent_complete"))
    assert len(of_type(events, "tool_call_start")) == len(of_type(events, "tool_call_end"))
    times = [e["time"] for e in events]
    assert times == sorted(times)


def test_models_and_reasoning(run):
    _, events = run
    models = {p["agent"]: p["model"] for p in of_type(events, "model_detected")}
    assert models["Crew Manager"] == "demo-gpt-4o" and models["Designer"] == "demo-gpt-4o-mini"
    roles = collections.Counter(p.get("role") for p in of_type(events, "message"))
    assert roles["thinking"] >= 5 and roles["assistant"] >= 5
