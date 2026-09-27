"""End-to-end test: the support desk demo through the trace processor."""
import asyncio
import collections
import json
import os
import sys

import pytest
from agents import set_trace_processors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import support_desk_demo as demo  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("oai") / "events.jsonl"
    # A small delay keeps parallel runs and tool calls in flight together.
    result = asyncio.run(demo.run(str(path), delay=0.02))
    set_trace_processors([])
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
    (support, mood), events = run
    assert support.startswith("Fixed and credited") and mood.startswith("Frustrated")
    assert events[-1]["type"] == "agent_complete" and events[-1]["payload"] == {"name": "Support desk"}


def test_agent_tree(run):
    _, events = run
    tree = parents(events)
    assert tree == {"Support desk": None, "Triage": "Support desk", "Sentiment Analyst": "Support desk",
                    "Tech Support": "Support desk", "Billing": "Support desk",
                    "Log Analyst": "Tech Support", "DB Inspector": "Log Analyst"}


def test_routing_graph(run):
    _, events = run
    assert hops(events, "Support desk") == {
        ("__start__", "Triage"): 1, ("__start__", "Sentiment Analyst"): 1,   # parallel runs
        ("Triage", "Tech Support"): 1, ("Tech Support", "Billing"): 1,      # handoffs
        ("Billing", "__end__"): 1, ("Sentiment Analyst", "__end__"): 1}
    declared = {(e["source"], e["target"]) for p in of_type(events, "graph_structure") for e in p["edges"]}
    assert ("Triage", "Sales") in declared and ("Triage", "Billing") in declared   # routes not taken


def test_parallel_tool_calls_overlap(run):
    _, events = run
    order = [(e["type"], e["payload"]["tool"]) for e in events
             if e["type"] in ("tool_call_start", "tool_call_end") and e["payload"]["agent"] == "Tech Support"]
    first_three = {t for kind, t in order[:3] if kind == "tool_call_start"}
    assert first_three == {"search_kb", "check_service_status", "analyze_logs"}


def test_failure_retry_and_guardrails(run):
    _, events = run
    status = [bool(p.get("isError")) for p in of_type(events, "tool_call_end") if p["tool"] == "check_service_status"]
    assert status == [True, False]
    checks = {p["tool"]: bool(p.get("isError")) for p in of_type(events, "tool_call_end") if p["tool"].startswith("guardrail")}
    assert checks == {"guardrail: pii_filter": False, "guardrail: refund_policy": False}


def test_everything_finishes(run):
    _, events = run
    started = collections.Counter((p["node"], p["step"]) for p in of_type(events, "node_start") if p["node"] != "__end__")
    ended = collections.Counter((p["node"], p["step"]) for p in of_type(events, "node_end"))
    assert started == ended
    assert collections.Counter(p["name"] for p in of_type(events, "agent_spawn")) == \
        collections.Counter(p["name"] for p in of_type(events, "agent_complete"))
    assert len(of_type(events, "tool_call_start")) == len(of_type(events, "tool_call_end"))


def test_models_and_reasoning(run):
    _, events = run
    models = {p["agent"]: p["model"] for p in of_type(events, "model_detected")}
    assert models["Tech Support"] == "demo-gpt-5" and models["DB Inspector"] == "demo-gpt-5-mini"
    roles = collections.Counter(p.get("role") for p in of_type(events, "message"))
    assert roles["thinking"] >= 3 and roles["user"] == 1
