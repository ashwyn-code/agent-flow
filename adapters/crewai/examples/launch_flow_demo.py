"""Nested, fan-out CrewAI demo for Agent Flow. No API key, no network.

    LaunchFlow (Flow)
    ├─ brief (@start)
    ├─▶ in parallel:
    │   ├─ research: research_crew
    │   │     market_scan  (async) ┐
    │   │     competitor_scan (async, a pricing lookup keeps failing, then the agent retries it)
    │   │                          ├─▶ insights (context = both)
    │   └─ creative: creative_crew, designer ─▶ brand_reviewer (sequential)
    ├─ draft (@listen(and_(research, creative))): copy_crew, hierarchical —
    │     the Crew Manager delegates to the copywriter and asks the editor
    ├─ review (@router): "revise" once, then "approve"
    ├─ polish (@listen("revise")): a single agent run, then back to review
    └─ publish (@listen("approve"))

Every CrewAI orchestration pattern in one run: Flow fan-out and and_ fan-in,
a router loop, crews with parallel async tasks and task context,
a hierarchical crew with delegation, and a direct agent kickoff.

    python examples/launch_flow_demo.py --out agent-flow.jsonl
"""

import argparse
import os
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keep CrewAI offline and quiet unless the caller already configured it
for key, value in {"OTEL_SDK_DISABLED": "true", "CREWAI_DISABLE_TELEMETRY": "true",
                   "CREWAI_DISABLE_TRACKING": "true", "CREWAI_TRACING_ENABLED": "false"}.items():
    os.environ.setdefault(key, value)
os.environ.setdefault("CREWAI_STORAGE_DIR", os.path.join(tempfile.gettempdir(), "agent-flow-crewai-demo"))

from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai.flow.flow import Flow, and_, listen, or_, router, start  # noqa: E402
from crewai.tools import tool  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_llm import answer, fake_llm, react  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_flow_crewai import AgentFlowListener  # noqa: E402

DELAY = 0.5
BRIEF = "Launch 'Pulse', our AI meeting summarizer, to mid-market teams next month"


def pause(key: str) -> None:
    if DELAY > 0:
        time.sleep(DELAY * (0.6 + (zlib.crc32(key.encode()) % 9) / 10))


def llm(policy: Any, model: str = "demo-gpt-4o-mini") -> Any:
    return fake_llm(policy, model=model, delay=lambda: DELAY * 0.4)


def member(role: str, policy: Any, tools: Optional[List[Any]] = None, model: str = "demo-gpt-4o-mini") -> Agent:
    return Agent(role=role, goal=f"Be an excellent {role}", backstory=f"A seasoned {role}.",
                 tools=tools or [], llm=llm(policy, model), verbose=False, allow_delegation=False)


# ─── Tools ───────────────────────────────────────────────────────────────────

_attempts: Dict[str, int] = {}
_lock = threading.Lock()


@tool("Web Search")
def web_search(query: str) -> str:
    """Search the web."""
    pause("search:" + query)
    return f"'{query}': 6 reports; mid-market AI tooling spend up 41% YoY"


@tool("Pricing Lookup")
def pricing_lookup(company: str) -> str:
    """Fetch a competitor's pricing page."""
    pause("pricing:" + company)
    with _lock:
        attempt = _attempts.get(company, 0)
        _attempts[company] = attempt + 1
    # Fails through CrewAI's three internal attempts, then works when the agent retries
    if company == "Otter" and attempt < 3:
        raise ConnectionError("pricing page returned 503")
    return f"{company}: $10 / $20 / enterprise per seat"


@tool("Image Generator")
def image_generator(prompt: str) -> str:
    """Generate a hero image."""
    pause("image:" + prompt)
    return f"hero image rendered for '{prompt}' (1920x1080)"


@tool("CMS Publish")
def cms_publish(slug: str) -> str:
    """Publish a page to the CMS."""
    pause("publish:" + slug)
    return f"published /{slug}"


# ─── Crews ───────────────────────────────────────────────────────────────────

def research_crew() -> Crew:
    market = member("Market Analyst", react(
        lambda task: [("Web Search", {"query": "mid-market meeting assistant adoption 2026"}),
                      ("Web Search", {"query": "AI summarizer willingness to pay"})],
        lambda task: "Mid-market spend on AI meeting tools is growing 41% a year.",
        thinking=lambda task: "Two searches: adoption, then pricing tolerance.",
    ), [web_search])
    competitor = member("Competitor Analyst", react(
        lambda task: [("Pricing Lookup", {"company": "Otter"}), ("Pricing Lookup", {"company": "Fireflies"})],
        lambda task: "Otter and Fireflies both price at $10-$20/seat; neither does action items well.",
    ), [pricing_lookup])
    lead = member("Insights Lead", answer("Position Pulse on action items at $15/seat.",
                                          "Combine market growth with the competitor gap."), model="demo-gpt-4o")
    market_scan = Task(name="market_scan", description="Size the mid-market opportunity for Pulse",
                       expected_output="market notes", agent=market, async_execution=True)
    competitor_scan = Task(name="competitor_scan", description="Compare competitor pricing for Pulse",
                           expected_output="pricing notes", agent=competitor, async_execution=True)
    insights = Task(name="insights", description="Turn the research into a positioning recommendation",
                    expected_output="positioning", agent=lead, context=[market_scan, competitor_scan])
    return Crew(name="research_crew", agents=[market, competitor, lead], tasks=[market_scan, competitor_scan, insights])


def creative_crew() -> Crew:
    designer = member("Designer", react(lambda task: [("Image Generator", {"prompt": "Pulse hero, calm meeting room"})],
                                        lambda task: "Hero image and palette ready."), [image_generator])
    reviewer = member("Brand Reviewer", answer("Assets match the brand guide."))
    return Crew(name="creative_crew", agents=[designer, reviewer], tasks=[
        Task(name="design_assets", description="Design the launch hero image", expected_output="assets", agent=designer),
        Task(name="brand_check", description="Check the assets against the brand guide", expected_output="verdict", agent=reviewer),
    ])


def copy_crew() -> Crew:
    copywriter = member("Copywriter", answer("Draft: 'Pulse turns every meeting into a to-do list.'",
                                             "Lead with the action-items angle."))
    editor = member("Editor", answer("Tighten the headline; keep the $15 price point."))
    manager = fake_llm(react(
        lambda task: [("Delegate work to coworker", {"task": "Write the launch blog post for Pulse",
                                                     "context": "Positioning: action items at $15/seat",
                                                     "coworker": "Copywriter"}),
                      ("Ask question to coworker", {"question": "What should we tighten before publishing?",
                                                    "context": "Launch blog post draft", "coworker": "Editor"})],
        lambda task: "Launch post drafted and edited.",
        thinking=lambda task: "Delegate the draft, then get an editor's pass.",
    ), model="demo-gpt-4o", delay=lambda: DELAY * 0.4)
    return Crew(name="copy_crew", agents=[copywriter, editor], process=Process.hierarchical, manager_llm=manager,
                tasks=[Task(name="launch_post", description="Produce the launch blog post", expected_output="blog post")])


# ─── Flow ────────────────────────────────────────────────────────────────────

class LaunchFlow(Flow):
    reviews: int = 0

    @start()
    def brief(self) -> str:
        pause("brief")
        return BRIEF

    @listen(brief)
    def research(self, brief: str) -> str:
        return str(research_crew().kickoff(inputs={"brief": brief}))

    @listen(brief)
    def creative(self, brief: str) -> str:
        return str(creative_crew().kickoff(inputs={"brief": brief}))

    @listen(and_(research, creative))
    def draft(self, _: Any) -> str:
        return str(copy_crew().kickoff())

    @router(or_(draft, "polished"), emit=["revise", "approve"])
    def review(self, _: Any) -> str:
        pause("review")
        LaunchFlow.reviews += 1
        return "revise" if LaunchFlow.reviews == 1 else "approve"

    @listen("revise")
    def polish(self, _: Any) -> str:
        editor = member("Line Editor", answer("Polished: headline shortened, CTA added."))
        return str(editor.kickoff("Polish the launch post's headline and call to action"))

    @listen(polish)
    def polished(self, _: Any) -> str:
        return "polished"

    @listen("approve")
    def publish(self, _: Any) -> str:
        publisher = member("Publisher", react(lambda task: [("CMS Publish", {"slug": "blog/introducing-pulse"})],
                                              lambda task: "Live at /blog/introducing-pulse."), [cms_publish])
        return str(publisher.kickoff("Publish the approved launch post"))


def run(out: str, delay: float = DELAY) -> Any:
    global DELAY
    DELAY = delay
    _attempts.clear()
    LaunchFlow.reviews = 0
    listener = AgentFlowListener(out, truncate=True)
    try:
        return LaunchFlow().kickoff()
    finally:
        listener.close()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="agent-flow-events.jsonl", help="JSONL file to write")
    parser.add_argument("--delay", type=float, default=DELAY, help="base seconds per tool / LLM call")
    args = parser.parse_args(argv)
    result = run(args.out, args.delay)
    print("result:", result)
    print(f"events written to {args.out}")


if __name__ == "__main__":
    main()
