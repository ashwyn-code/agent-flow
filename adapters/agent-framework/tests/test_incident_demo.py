"""End-to-end test: the incident-response demo workflow through the adapter."""
import asyncio
import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import incident_response_demo as demo  # noqa: E402

ANALYSTS = ["log_analyst", "metrics_analyst", "deploy_auditor"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("maf") / "events.jsonl"
    # A small delay keeps the parallel analysts in flight together.
    result = asyncio.run(demo.run(str(path), delay=0.02))
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
    assert result.get_outputs() == ["Incident closed. Postmortem drafted: pool-size regression in v2.41.0, 38 minutes of impact."]
    assert events[-1]["type"] == "agent_complete" and events[-1]["payload"] == {"name": "incident_response"}


def test_agent_tree(run):
    _, events = run
    tree = parents(events)
    assert len(tree) == 12
    assert tree["incident_response"] is None
    assert {n for n, p in tree.items() if p == "incident_response"} == {
        "triage", *ANALYSTS, "commander", "remediation", "postmortem"}
    assert tree["git_historian"] == "deploy_auditor"          # agent.as_tool()
    assert {n for n, p in tree.items() if p == "remediation"} == {"planner", "operator"}
    assert tree["k8s_bot"] == "operator"                      # as_tool inside a nested workflow

    def depth(name):
        return 0 if tree[name] is None else 1 + depth(tree[name])
    assert max(depth(n) for n in tree) == 3


def test_analysts_run_in_parallel(run):
    _, events = run
    order = [(e["type"], e["payload"].get("child")) for e in events if e["type"] in ("subagent_dispatch", "subagent_return")]
    started = [i for i, (kind, child) in enumerate(order) if kind == "subagent_dispatch" and child in ANALYSTS]
    first_done = min(i for i, (kind, child) in enumerate(order) if kind == "subagent_return" and child in ANALYSTS)
    assert len(started) == 3 and max(started) < first_done


def test_structure(run):
    _, events = run
    structures = {p["agent"]: p for p in of_type(events, "graph_structure")}
    assert set(structures) == {"incident_response", "remediation"}
    kinds = {n["id"]: n["kind"] for n in structures["incident_response"]["nodes"]}
    assert kinds["remediation"] == "subgraph" and kinds["triage"] == "node"
    conditional = {(e["source"], e["target"]) for e in structures["incident_response"]["edges"] if e["conditional"]}
    assert conditional == {("triage", "dispatch"), ("triage", "auto_ack"),       # switch-case
                           ("remediation", "commander"), ("remediation", "postmortem")}


def test_routing_fan_out_fan_in_and_loop(run):
    _, events = run
    h = hops(events, "incident_response")
    assert h[("triage", "dispatch")] == 1 and h[("triage", "auto_ack")] == 0   # switch-case took the high branch
    for analyst in ANALYSTS:
        assert h[("dispatch", analyst)] == 1 and h[(analyst, "correlate")] == 1
    assert h[("commander", "remediation")] == 2
    assert h[("remediation", "commander")] == 1   # the FAIL loop, taken once
    assert h[("remediation", "postmortem")] == 1
    assert h[("publish", "__end__")] == 1
    correlate = next(p for p in of_type(events, "node_start") if p["node"] == "correlate")
    assert correlate["from"] == sorted(ANALYSTS)


def test_nested_workflow_runs_each_remediation(run):
    _, events = run
    assert hops(events, "remediation") == {
        ("__start__", "planner"): 2, ("planner", "operator"): 2, ("operator", "verifier"): 2, ("verifier", "__end__"): 2}


def test_tool_failure_and_retry(run):
    _, events = run
    ends = [p for p in of_type(events, "tool_call_end") if p["agent"] == "metrics_analyst"]
    assert [(p["tool"], bool(p.get("isError"))) for p in ends] == [
        ("query_metrics", True), ("query_metrics", False), ("query_metrics", False)]
    assert "timed out" in ends[0]["errorMessage"]


def test_everything_finishes(run):
    _, events = run
    started = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_start") if p["node"] != "__end__")
    ended = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_end"))
    assert started == ended
    assert collections.Counter(p["name"] for p in of_type(events, "agent_spawn")) == \
        collections.Counter(p["name"] for p in of_type(events, "agent_complete"))
    assert collections.Counter(p["child"] for p in of_type(events, "subagent_dispatch")) == \
        collections.Counter(p["child"] for p in of_type(events, "subagent_return"))
    assert len(of_type(events, "tool_call_start")) == len(of_type(events, "tool_call_end"))


def test_models_and_reasoning(run):
    _, events = run
    models = {p["agent"]: p["model"] for p in of_type(events, "model_detected")}
    assert models["commander"] == "demo-gpt-5" and models["k8s_bot"] == "demo-gpt-5-mini"
    roles = collections.Counter(p.get("role") for p in of_type(events, "message"))
    assert roles["thinking"] >= 4 and roles["user"] == 1
