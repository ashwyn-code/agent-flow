"""End-to-end test: the nested fan-out Strands demo through the hooks."""
import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import orchestration_demo as demo  # noqa: E402

BRANCHES = ["market_research", "competitor_scan", "tech_review"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("strands") / "events.jsonl"
    # A small delay keeps the parallel branches and analysts in flight together.
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
    assert result.status.value == "completed"
    assert events[-1]["type"] == "agent_complete" and events[-1]["payload"] == {"name": "product_launch"}


def test_agent_tree(run):
    _, events = run
    tree = parents(events)
    assert len(tree) == 19
    assert tree["product_launch"] is None
    assert {n for n, p in tree.items() if p == "product_launch"} == {
        "planner", *BRANCHES, "strategist", "reviewer", "publisher"}
    assert {n for n, p in tree.items() if p == "market_research"} == {"collector", "survey_analyst", "trend_analyst", "synthesizer"}
    assert tree["statistician"] == "survey_analyst"  # agent as tool inside a nested graph
    assert {n for n, p in tree.items() if p == "competitor_scan"} == {"analyst_acme", "analyst_globex", "analyst_initech"}
    assert {n for n, p in tree.items() if p == "tech_review"} == {"architect", "security", "performance"}

    def depth(name):
        return 0 if tree[name] is None else 1 + depth(tree[name])
    assert max(depth(n) for n in tree) == 3


def test_branches_and_analysts_run_in_parallel(run):
    _, events = run
    order = [(e["type"], e["payload"].get("child")) for e in events if e["type"] in ("subagent_dispatch", "subagent_return")]

    def overlapping(names):
        started = [i for i, (kind, child) in enumerate(order) if kind == "subagent_dispatch" and child in names]
        first_done = min(i for i, (kind, child) in enumerate(order) if kind == "subagent_return" and child in names)
        return len(started) >= len(names) and sorted(started)[len(names) - 1] < first_done

    assert overlapping(BRANCHES)
    assert overlapping(["analyst_acme", "analyst_globex", "analyst_initech"])


def test_graph_structures(run):
    _, events = run
    structures = {p["agent"]: p for p in of_type(events, "graph_structure")}
    assert set(structures) == {"product_launch", "market_research", "tech_review"}
    kinds = {n["id"]: n["kind"] for n in structures["product_launch"]["nodes"]}
    assert kinds["market_research"] == "subgraph" and kinds["tech_review"] == "subgraph"
    assert kinds["competitor_scan"] == "node"
    conditional = {(e["source"], e["target"]) for e in structures["product_launch"]["edges"] if e["conditional"]}
    assert conditional == {("reviewer", "strategist"), ("reviewer", "publisher")}


def test_top_level_fan_out_merge_and_review_loop(run):
    _, events = run
    h = hops(events, "product_launch")
    for branch in BRANCHES:
        assert h[("planner", branch)] == 1 and h[(branch, "strategist")] == 1
    assert h[("strategist", "reviewer")] == 2
    assert h[("reviewer", "strategist")] == 1
    assert h[("reviewer", "publisher")] == 1
    assert h[("publisher", "__end__")] == 1


def test_nested_graph_fan_out_and_merge(run):
    _, events = run
    assert hops(events, "market_research") == {
        ("__start__", "collector"): 1, ("collector", "survey_analyst"): 1, ("collector", "trend_analyst"): 1,
        ("survey_analyst", "synthesizer"): 1, ("trend_analyst", "synthesizer"): 1, ("synthesizer", "__end__"): 1}


def test_swarm_handoffs(run):
    _, events = run
    assert hops(events, "tech_review") == {
        ("__start__", "architect"): 1, ("architect", "security"): 1, ("security", "architect"): 1,
        ("architect", "performance"): 1, ("performance", "__end__"): 1}


def test_tool_failure_and_retry(run):
    _, events = run
    ends = of_type(events, "tool_call_end")
    failures = [p for p in ends if p.get("isError")]
    assert [(p["agent"], p["tool"]) for p in failures] == [("analyst_globex", "pricing_page")]
    assert "503" in failures[0]["errorMessage"]
    retries = [bool(p.get("isError")) for p in ends if p["agent"] == "analyst_globex" and p["tool"] == "pricing_page"]
    assert retries == [True, False]


def test_everything_finishes(run):
    _, events = run
    started = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_start") if p["node"] != "__end__")
    ended = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_end"))
    assert started == ended
    # (a nested graph re-sends its spawn once its task is known, so compare names)
    assert {p["name"] for p in of_type(events, "agent_spawn")} == {p["name"] for p in of_type(events, "agent_complete")}
    dispatched = collections.Counter(p["child"] for p in of_type(events, "subagent_dispatch"))
    returned = collections.Counter(p["child"] for p in of_type(events, "subagent_return"))
    assert dispatched == returned
    assert len(of_type(events, "tool_call_start")) == len(of_type(events, "tool_call_end"))


def test_models_and_reasoning(run):
    _, events = run
    models = {p["agent"]: p["model"] for p in of_type(events, "model_detected")}
    assert models["planner"] == "demo-claude-opus-4-1"
    assert models["analyst_acme"] == "demo-claude-sonnet-4-5"
    roles = collections.Counter(p.get("role") for p in of_type(events, "message"))
    assert roles["thinking"] >= 5 and roles["user"] == 1
