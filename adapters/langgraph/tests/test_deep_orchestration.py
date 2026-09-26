"""End-to-end test: the deeply nested, fan-out orchestration demo through the adapter."""
import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import deep_orchestration_demo as deep  # noqa: E402

TEAMS = ["research_team", "research_team #2", "research_team #3"]
SPECIALISTS = ["web_researcher", "paper_researcher", "data_analyst"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("deep") / "events.jsonl"
    # A small delay keeps the three parallel teams in flight at the same time.
    result = deep.run(str(path), delay=0.01)
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    return result, events


def of_type(events, kind):
    return [e["payload"] for e in events if e["type"] == kind]


def parents(events):
    return {p["name"]: p.get("parent") for p in of_type(events, "agent_spawn")}


def hops(events, agent):
    return collections.Counter(
        (src, p["node"]) for p in of_type(events, "node_start") if p["agent"] == agent for src in p["from"])


def test_run_completes_and_publishes(run):
    result, events = run
    assert result["verdict"] == "APPROVE"
    assert result["revisions"] == 2
    assert result["messages"][-1].content.startswith("Published")
    assert events[-1] == {**events[-1], "type": "agent_complete", "payload": {"name": "orchestrator"}}


def test_agent_tree_is_four_levels_deep_with_seventeen_agents(run):
    _, events = run
    tree = parents(events)
    assert len(tree) == 17  # 1 + 3 teams + 9 specialists + writer, drafter, section_writer, fact_checker
    assert tree["orchestrator"] is None
    for team in TEAMS:
        assert tree[team] == "orchestrator"
    # Every team has its own three specialists
    team_children = collections.defaultdict(set)
    for name, parent in tree.items():
        if parent in TEAMS:
            team_children[parent].add(name.split(" #")[0])
    assert team_children == {team: set(SPECIALISTS) for team in TEAMS}
    assert tree["writer"] == "orchestrator"
    assert tree["drafter"] == "writer"
    assert tree["section_writer"] == "drafter"
    assert tree["fact_checker"] == "drafter"

    def depth(name):
        return 0 if tree[name] is None else 1 + depth(tree[name])
    assert max(depth(n) for n in tree) == 3  # orchestrator → writer → drafter → section_writer


def test_research_teams_run_in_parallel(run):
    _, events = run
    # All three teams are dispatched before any of them returns.
    order = [(e["type"], e["payload"].get("child")) for e in events if e["type"] in ("subagent_dispatch", "subagent_return")]
    dispatched = [i for i, (kind, child) in enumerate(order) if kind == "subagent_dispatch" and child in TEAMS]
    first_return = next(i for i, (kind, child) in enumerate(order) if kind == "subagent_return" and child in TEAMS)
    assert len(dispatched) == 3 and max(dispatched) < first_return


def test_every_graph_sends_its_structure(run):
    _, events = run
    structures = {p["agent"]: p for p in of_type(events, "graph_structure")}
    assert set(structures) == set(parents(events))
    main = {n["id"]: n["kind"] for n in structures["orchestrator"]["nodes"]}
    assert main["research_team"] == "subgraph" and main["writer"] == "subgraph"
    team = {n["id"]: n["kind"] for n in structures["research_team #2"]["nodes"]}
    assert all(team[s] == "subgraph" for s in SPECIALISTS)
    assert {n["id"] for n in structures["drafter"]["nodes"]} >= {"section_writer", "fact_checker", "compose"}
    critic_edges = {(e["source"], e["target"]): e["conditional"] for e in structures["orchestrator"]["edges"] if e["source"] == "critic"}
    assert critic_edges == {("critic", "writer"): True, ("critic", "publish"): True}


def test_orchestrator_fan_out_merge_and_review_loop(run):
    _, events = run
    h = hops(events, "orchestrator")
    assert h[("planner", "research_team")] == 3      # Send fan-out
    assert h[("research_team", "synthesize")] == 3   # fan-in of all teams
    assert h[("critic", "writer")] == 1              # critic sends the report back once
    assert h[("writer", "critic")] == 2
    assert h[("critic", "publish")] == 1
    assert h[("publish", "__end__")] == 1


def test_team_fan_out_and_merge(run):
    _, events = run
    for team in TEAMS:
        h = hops(events, team)
        for s in SPECIALISTS:
            assert h[("lead", s)] == 1 and h[(s, "merge")] == 1
        # The merge node runs once, after all three specialists
        merges = [p for p in of_type(events, "node_start") if p["agent"] == team and p["node"] == "merge"]
        assert len(merges) == 1 and sorted(merges[0]["from"]) == sorted(SPECIALISTS)


def test_nested_loops_inside_writer(run):
    _, events = run
    writer = hops(events, "writer")
    assert writer[("editor", "drafter")] == 2  # one editor loop per writer pass
    assert writer[("drafter", "editor")] == 4
    drafter = hops(events, "drafter")
    assert drafter[("section_writer", "fact_checker")] == 4
    react = hops(events, "section_writer")
    # Parallel tool calls run as separate `tools` tasks: 3 sections × 4 drafts
    assert react[("agent", "tools")] == 12 and react[("tools", "agent")] == 12


def test_parallel_tool_calls_and_retry_after_failure(run):
    _, events = run
    starts = of_type(events, "tool_call_start")
    ends = of_type(events, "tool_call_end")
    assert len(starts) == len(ends)
    sections = [p for p in starts if p["tool"] == "write_section"]
    assert len(sections) == 12  # 3 parallel sections × 4 drafts

    failures = [p for p in ends if p.get("isError")]
    assert [(p["tool"], "timed out" in p["errorMessage"]) for p in failures] == [("run_sql", True)]
    analyst = failures[0]["agent"]
    runs = [p for p in ends if p["agent"] == analyst and p["tool"] == "run_sql"]
    assert [bool(p.get("isError")) for p in runs] == [True, False]  # retried and succeeded
    assert parents(events)[analyst] in TEAMS


def test_every_node_and_agent_finishes(run):
    _, events = run
    started = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_start") if p["node"] != "__end__")
    ended = collections.Counter((p["agent"], p["node"], p["step"]) for p in of_type(events, "node_end"))
    assert started == ended
    completed = collections.Counter(p["name"] for p in of_type(events, "agent_complete"))
    spawned = collections.Counter(p["name"] for p in of_type(events, "agent_spawn"))
    assert completed == spawned
    returns = collections.Counter(p["child"] for p in of_type(events, "subagent_return"))
    dispatches = collections.Counter(p["child"] for p in of_type(events, "subagent_dispatch"))
    assert returns == dispatches


def test_models_thinking_and_context_are_reported(run):
    _, events = run
    models = {p["agent"]: p["model"] for p in of_type(events, "model_detected")}
    assert models["orchestrator"] == "demo-opus-4-1"
    assert models["web_researcher"] == "demo-sonnet-4-5"
    roles = collections.Counter(p.get("role") for p in of_type(events, "message"))
    assert roles["thinking"] >= 10 and roles["user"] == 1
    assert all(p["tokens"] > 0 for p in of_type(events, "context_update"))
