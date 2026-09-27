"""Nested, fan-out Strands multi-agent demo for Agent Flow. No API key.

    product_launch (Graph)
    ├─ planner
    ├─▶ in parallel:
    │   ├─ market_research (nested Graph)
    │   │    collector (2 parallel tool calls) ─▶ survey_analyst ┐
    │   │                                     ─▶ trend_analyst  ├─▶ synthesizer
    │   │    survey_analyst asks a statistician agent (agent as tool)
    │   ├─ competitor_scan: 3 parallel tool calls, each running its own
    │   │    analyst agent (agents as tools); one pricing fetch fails and is retried
    │   └─ tech_review (Swarm): architect ─▶ security ─▶ architect ─▶ performance
    ├─ strategist (fan-in of all three)
    ├─ reviewer ──REVISE──▶ strategist (loop once) / ──APPROVE──▶ publisher
    └─ publisher

Every Strands multi-agent pattern in one run: a Graph with parallel batches,
a conditional loop and a nested Graph, a Swarm with handoffs, and agents as
tools, four levels deep with 19 agents.

    python examples/orchestration_demo.py --out agent-flow.jsonl
"""

import argparse
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from strands import Agent, tool
from strands.multiagent import GraphBuilder, Swarm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_model import RuleBasedModel, react, tool_results  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_flow_strands import AgentFlowHooks  # noqa: E402

DELAY = 0.5
REQUEST = "Plan the launch of our AI analytics product: market, competitors, architecture and a go/no-go strategy"
COMPETITORS = ["acme", "globex", "initech"]


def pause(key: str) -> None:
    if DELAY > 0:
        time.sleep(DELAY * (0.6 + (zlib.crc32(key.encode()) % 9) / 10))


def model(policy: Any, model_id: str = "demo-claude-sonnet-4-5") -> RuleBasedModel:
    return RuleBasedModel(policy, model_id=model_id, delay=lambda: DELAY * 0.4)


def agent(name: str, policy: Any, tools: Optional[List[Any]] = None, hooks: Optional[List[Any]] = None, **kw: Any) -> Agent:
    return Agent(name=name, model=model(policy, **kw), tools=tools or [], hooks=hooks or [], callback_handler=None)


def say(text: str, thinking: Optional[str] = None) -> Any:
    def policy(messages: Any, names: Any) -> Dict[str, Any]:
        reply: Dict[str, Any] = {"text": text}
        if thinking:
            reply["thinking"] = thinking
        return reply
    return policy


# ─── Tools ───────────────────────────────────────────────────────────────────

_pricing_attempts: Dict[str, int] = {}
_lock = threading.Lock()


@tool
def fetch_dataset(name: str) -> str:
    """Load a market dataset."""
    pause("dataset:" + name)
    return f"dataset {name}: 18k rows, 2023-2026"


@tool
def search_trends(query: str) -> str:
    """Search trend reports."""
    pause("trends:" + query)
    return f"trend reports on {query}: adoption up 34% YoY"


@tool
def run_regression(spec: str) -> str:
    """Run a regression."""
    pause("regression:" + spec)
    return f"regression {spec}: r2=0.71, willingness to pay rises with team size"


@tool
def web_lookup(company: str) -> str:
    """Look a company up."""
    pause("lookup:" + company)
    return f"{company}: 240 employees, series C, 3 analytics products"


@tool
def pricing_page(company: str) -> str:
    """Scrape a company's pricing page."""
    pause("pricing:" + company)
    with _lock:
        attempt = _pricing_attempts.get(company, 0)
        _pricing_attempts[company] = attempt + 1
    if company == "globex" and attempt == 0:
        raise ConnectionError("pricing page returned 503")
    return f"{company} pricing: $49 / $199 / enterprise"


@tool
def publish_report(channel: str) -> str:
    """Publish the launch plan."""
    pause("publish:" + channel)
    return f"published to {channel}"


# ─── Graph pieces ────────────────────────────────────────────────────────────

def build(flow: AgentFlowHooks):
    statistician = agent("statistician", react(
        lambda task: [[("run_regression", {"spec": "wtp ~ team_size + industry"})]],
        lambda task, results: "Willingness to pay grows with team size (r2=0.71).",
    ), [run_regression], hooks=[flow])

    @tool
    def ask_statistician(question: str) -> str:
        """Ask the statistician agent a question."""
        return str(statistician(question)).strip()

    # market_research: a nested graph with its own fan-out and fan-in
    collector = agent("collector", react(
        lambda task: [[("fetch_dataset", {"name": "survey_2026"}), ("fetch_dataset", {"name": "usage_telemetry"})]],
        lambda task, results: f"Collected {len(results)} datasets.",
        thinking=lambda task: "Pull both datasets at once.",
    ), [fetch_dataset])
    survey_analyst = agent("survey_analyst", react(
        lambda task: [[("ask_statistician", {"question": "What drives willingness to pay?"})]],
        lambda task, results: "Survey: teams of 20+ convert at 3x; pricing sensitivity is low.",
    ), [ask_statistician])
    trend_analyst = agent("trend_analyst", react(
        lambda task: [[("search_trends", {"query": "AI analytics adoption"})]],
        lambda task, results: "Trends: category growing 34% YoY; buyers want AI summaries.",
    ), [search_trends])
    synthesizer = agent("synthesizer", say("Market: $4.2B, growing 34%, sweet spot is 20-200 seat teams."))

    mb = GraphBuilder()
    for name, node in (("collector", collector), ("survey_analyst", survey_analyst),
                       ("trend_analyst", trend_analyst), ("synthesizer", synthesizer)):
        mb.add_node(node, name)
    mb.add_edge("collector", "survey_analyst")
    mb.add_edge("collector", "trend_analyst")
    mb.add_edge("survey_analyst", "synthesizer")
    mb.add_edge("trend_analyst", "synthesizer")
    mb.set_graph_id("market_research")
    market_research = mb.build()

    # competitor_scan: agents as tools, three in parallel
    @tool
    def analyze_competitor(company: str) -> str:
        """Run a dedicated analyst agent on one competitor."""
        analyst = agent(f"analyst_{company}", react(
            lambda task: [[("web_lookup", {"company": company}), ("pricing_page", {"company": company})]],
            lambda task, results: f"{company}: strong in SMB, weak on AI features, priced at $49-$199.",
            thinking=lambda task: f"Profile {company}: company facts and pricing in parallel.",
        ), [web_lookup, pricing_page], hooks=[flow])
        return str(analyst(f"Analyse competitor {company}")).strip()

    competitor_scan = agent("competitor_scan", react(
        lambda task: [[("analyze_competitor", {"company": c}) for c in COMPETITORS]],
        lambda task, results: f"Scanned {len(results)} competitors; none ship AI summaries yet.",
        thinking=lambda task: "One analyst per competitor, all at once.",
    ), [analyze_competitor])

    # tech_review: a swarm handing work around. The architect hands to
    # security first, gets it back, then hands to performance.
    architect_visits: List[int] = []

    def architect_policy(messages: Any, names: Any) -> Dict[str, Any]:
        if tool_results(messages, ["handoff_to_agent"]):
            return {"text": "Architecture handed off."}
        architect_visits.append(1)
        target = "security" if len(architect_visits) == 1 else "performance"
        return {"thinking": f"Next: {target} review.",
                "tools": [("handoff_to_agent", {"agent_name": target, "message": f"Please review {target}"})]}

    def security_policy(messages: Any, names: Any) -> Dict[str, Any]:
        if tool_results(messages, ["handoff_to_agent"]):
            return {"text": "Security notes passed back."}
        return {"text": "Found 2 issues in token storage.",
                "tools": [("handoff_to_agent", {"agent_name": "architect", "message": "Fix token storage, then continue"})]}

    tech_review = Swarm(
        [agent("architect", architect_policy), agent("security", security_policy),
         agent("performance", say("p95 latency 180ms under 5k rps: ship it."))],
        max_handoffs=10, max_iterations=10,
    )
    tech_review.id = "tech_review"

    # Top-level graph with a review loop
    reviews: List[str] = []

    def reviewer_policy(messages: Any, names: Any) -> Dict[str, Any]:
        verdict = "REVISE: quantify the pricing risk." if not reviews else "APPROVE: go for launch."
        reviews.append(verdict)
        return {"text": verdict, "thinking": "Check the plan against market, competition and tech findings."}

    def needs_revision(state: Any) -> bool:
        result = state.results.get("reviewer")
        return result is not None and "REVISE" in str(result.result)

    def approved(state: Any) -> bool:
        result = state.results.get("reviewer")
        return result is not None and "APPROVE" in str(result.result)

    gb = GraphBuilder()
    gb.add_node(agent("planner", say("Research market, competitors and architecture in parallel, then decide.",
                                     thinking="Three independent threads; run them concurrently."),
                      model_id="demo-claude-opus-4-1"), "planner")
    gb.add_node(market_research, "market_research")
    gb.add_node(competitor_scan, "competitor_scan")
    gb.add_node(tech_review, "tech_review")
    gb.add_node(agent("strategist", say("Strategy: launch to 20-200 seat teams at $99/seat with AI summaries."),
                      model_id="demo-claude-opus-4-1"), "strategist")
    gb.add_node(agent("reviewer", reviewer_policy, model_id="demo-claude-opus-4-1"), "reviewer")
    gb.add_node(agent("publisher", react(lambda task: [[("publish_report", {"channel": "exec-updates"})]],
                                         lambda task, results: "Launch plan published."),
                      [publish_report]), "publisher")
    for branch in ("market_research", "competitor_scan", "tech_review"):
        gb.add_edge("planner", branch)
        gb.add_edge(branch, "strategist")
    gb.add_edge("strategist", "reviewer")
    gb.add_edge("reviewer", "strategist", condition=needs_revision)
    gb.add_edge("reviewer", "publisher", condition=approved)
    gb.set_entry_point("planner")
    gb.reset_on_revisit(True)
    gb.set_max_node_executions(40)
    gb.set_graph_id("product_launch")
    return flow.instrument(gb.build())


def run(out: str, delay: float = DELAY) -> Any:
    global DELAY
    DELAY = delay
    _pricing_attempts.clear()
    flow = AgentFlowHooks(out, truncate=True)
    graph = build(flow)
    try:
        return graph(REQUEST)
    finally:
        flow.flush()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="agent-flow-events.jsonl", help="JSONL file to write")
    parser.add_argument("--delay", type=float, default=DELAY, help="base seconds per tool / model call")
    args = parser.parse_args(argv)
    result = run(args.out, args.delay)
    print(f"status: {result.status.value}; {result.completed_nodes}/{result.total_nodes} nodes completed")
    print(f"events written to {args.out}")


if __name__ == "__main__":
    main()
