import asyncio
import collections
import json
import os
import sys

import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    Runner,
    function_tool,
    output_guardrail,
    set_trace_processors,
    trace,
)
from agents.tracing import get_trace_provider

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
sys.path.insert(0, os.path.join(HERE, ".."))
from fake_model import RuleBasedModel, react  # noqa: E402

from agent_flow_openai_agents import AgentFlowProcessor, install  # noqa: E402

# Event types the webview understands (web/hooks/simulation/process-event.ts).
KNOWN_TYPES = {
    "agent_spawn", "agent_complete", "agent_idle", "message", "context_update", "model_detected",
    "tool_call_start", "tool_call_end", "subagent_dispatch", "subagent_return", "permission_requested",
    "graph_structure", "node_start", "node_end",
}


@pytest.fixture
def processor(tmp_path):
    processor = install(str(tmp_path / "events.jsonl"), exclusive=True)
    yield processor
    set_trace_processors([])


def events_of(processor):
    processor.flush()
    if not os.path.exists(processor.path):
        return []
    with open(processor.path, encoding="utf-8") as f:
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
    return lambda items: {"text": text}


@function_tool
async def lookup(q: str) -> str:
    """Look something up."""
    return f"found {q}"


_flaky_calls = []


@function_tool
async def flaky(q: str) -> str:
    """Fails the first time."""
    _flaky_calls.append(q)
    if len(_flaky_calls) == 1:
        raise RuntimeError("upstream exploded")
    return "worked"


def helper(name="Helper", model_name="demo-model"):
    policy = react(lambda task: [[("lookup", {"q": "x"}), ("lookup", {"q": "y"})]], lambda task, results: "all done",
                   thinking=lambda task: "two lookups")
    return Agent(name=name, instructions="help", tools=[lookup], model=RuleBasedModel(policy, name=model_name))


@pytest.mark.parametrize("mode", ["run", "run_sync", "run_streamed"])
def test_single_agent_events(processor, mode):
    agent = helper()
    if mode == "run":
        result = asyncio.run(Runner.run(agent, "find things"))
    elif mode == "run_sync":
        result = Runner.run_sync(agent, "find things")
    else:
        async def streamed():
            stream = Runner.run_streamed(agent, "find things")
            async for _ in stream.stream_events():
                pass
            return stream
        result = asyncio.run(streamed())
    assert result.final_output == "all done"
    events = events_of(processor)
    assert all(e["type"] in KNOWN_TYPES for e in events)
    assert parents(events) == {"Agent workflow": None, "Helper": "Agent workflow"}
    assert of_type(events, "message")[0] == {"agent": "Agent workflow", "role": "user", "content": "find things"}
    assert of_type(events, "model_detected") == [{"agent": "Helper", "model": "demo-model"}]
    starts = of_type(events, "tool_call_start")
    assert sorted((p["agent"], p["tool"], p["inputData"]["q"]) for p in starts) == [("Helper", "lookup", "x"), ("Helper", "lookup", "y")]
    assert sorted(p["result"] for p in of_type(events, "tool_call_end")) == ["found x", "found y"]
    roles = [(p.get("role"), p["agent"]) for p in of_type(events, "message")]
    assert ("thinking", "Helper") in roles and ("assistant", "Helper") in roles
    assert hops(events, "Agent workflow") == {("__start__", "Helper"): 1, ("Helper", "__end__"): 1}
    assert events[-1] == {**events[-1], "type": "agent_complete", "payload": {"name": "Agent workflow"}}


def test_handoffs_form_a_routing_graph(processor):
    billing = Agent(name="Billing", instructions="b", model=RuleBasedModel(say("refunded")))
    sales = Agent(name="Sales", instructions="s", model=RuleBasedModel(say("sold")))
    triage = Agent(name="Triage", instructions="t", handoffs=[billing, sales],
                   model=RuleBasedModel(react(lambda task: [[("transfer_to_billing", {})]], lambda task, r: "routed")))
    assert asyncio.run(Runner.run(triage, "refund please")).final_output == "refunded"
    events = events_of(processor)
    assert parents(events) == {"Agent workflow": None, "Triage": "Agent workflow", "Billing": "Agent workflow"}
    assert hops(events, "Agent workflow") == {("__start__", "Triage"): 1, ("Triage", "Billing"): 1, ("Billing", "__end__"): 1}
    declared = {(e["source"], e["target"]): e["conditional"] for p in of_type(events, "graph_structure") for e in p["edges"]}
    assert declared == {("Triage", "Billing"): True, ("Triage", "Sales"): True}   # Sales declared, never taken
    tools = [(p["agent"], p["tool"]) for p in of_type(events, "tool_call_start")]
    assert tools == [("Triage", "handoff → Billing")]


def test_agents_as_tools_become_subagents_even_concurrently(processor):
    researcher = Agent(name="Researcher", instructions="r", tools=[lookup], model=RuleBasedModel(react(
        lambda task: [[("lookup", {"q": task})]], lambda task, results: f"notes on {task}")))
    boss = Agent(name="Boss", instructions="b", tools=[researcher.as_tool(tool_name="research", tool_description="research")],
                 model=RuleBasedModel(react(lambda task: [[("research", {"input": t}) for t in ("a", "b", "c")]],
                                            lambda task, results: "merged")))
    asyncio.run(Runner.run(boss, "plan"))
    events = events_of(processor)
    tree = parents(events)
    assert sorted(n for n, p in tree.items() if p == "Boss") == ["Researcher", "Researcher #2", "Researcher #3"]
    assert sorted(p["summary"] for p in of_type(events, "subagent_return") if p["parent"] == "Boss") == [
        "notes on a", "notes on b", "notes on c"]
    inner = sorted(p["agent"] for p in of_type(events, "tool_call_start") if p["tool"] == "lookup")
    assert inner == ["Researcher", "Researcher #2", "Researcher #3"]
    # The nested runs are not nodes of the routing graph
    assert hops(events, "Agent workflow") == {("__start__", "Boss"): 1, ("Boss", "__end__"): 1}


def test_parallel_runs_in_one_trace_branch_from_start(processor):
    async def go():
        with trace("Desk"):
            await asyncio.gather(Runner.run(helper("Left"), "l"), Runner.run(helper("Right"), "r"))
    asyncio.run(go())
    events = events_of(processor)
    assert parents(events) == {"Desk": None, "Left": "Desk", "Right": "Desk"}
    assert hops(events, "Desk") == {("__start__", "Left"): 1, ("__start__", "Right"): 1,
                                    ("Left", "__end__"): 1, ("Right", "__end__"): 1}


def test_separate_runs_are_separate_traces(processor):
    asyncio.run(Runner.run(helper(), "one"))
    asyncio.run(Runner.run(helper(), "two"))
    mains = [p["name"] for p in of_type(events_of(processor), "agent_spawn") if p.get("isMain")]
    assert mains == ["Agent workflow", "Agent workflow"]


def test_tool_error_is_reported_and_retried(processor):
    _flaky_calls.clear()
    agent = Agent(name="Worker", instructions="w", tools=[flaky],
                  model=RuleBasedModel(react(lambda task: [[("flaky", {"q": "x"})]], lambda task, results: "done")))
    asyncio.run(Runner.run(agent, "go"))
    ends = of_type(events_of(processor), "tool_call_end")
    assert [(p["tool"], bool(p.get("isError"))) for p in ends] == [("flaky", True), ("flaky", False)]
    assert "upstream exploded" in ends[0]["errorMessage"]


def test_guardrail_tripwire_is_a_failed_check(processor):
    @output_guardrail
    async def no_refunds(ctx, agent, output):
        return GuardrailFunctionOutput(output_info={}, tripwire_triggered=True)

    agent = Agent(name="Clerk", instructions="c", output_guardrails=[no_refunds], model=RuleBasedModel(say("refund issued")))
    with pytest.raises(OutputGuardrailTripwireTriggered):
        asyncio.run(Runner.run(agent, "refund"))
    events = events_of(processor)
    check = next(p for p in of_type(events, "tool_call_end") if p["tool"] == "guardrail: no_refunds")
    assert check["isError"] is True
    end = next(p for p in of_type(events, "node_end") if p["node"] == "Clerk")
    assert "Guardrail tripwire triggered" in end["error"]
    assert events[-1]["type"] == "agent_complete"


def test_install_modes(tmp_path):
    exclusive = install(str(tmp_path / "a.jsonl"), exclusive=True)
    assert get_trace_provider()._multi_processor._processors == (exclusive,)
    added = install(str(tmp_path / "b.jsonl"))
    assert get_trace_provider()._multi_processor._processors == (exclusive, added)
    set_trace_processors([])


def test_a_failing_writer_never_breaks_the_run(processor, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(processor, "_emit", boom)
    assert asyncio.run(Runner.run(helper(), "go")).final_output == "all done"


def test_processor_is_a_tracing_processor():
    from agents.tracing import TracingProcessor
    assert issubclass(AgentFlowProcessor, TracingProcessor)
