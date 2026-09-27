"""Deeply nested, fan-out multi-agent LangGraph demo for Agent Flow. No API key.

    orchestrator (main graph)
    ├─ planner ──Send×3──▶ research_team (×3 in parallel: one per topic)
    │     research_team: lead ─▶ web_researcher   ┐
    │                         ─▶ paper_researcher ├─▶ merge     (fan-out / fan-in)
    │                         ─▶ data_analyst     ┘
    │     each specialist is a ReAct agent (agent ⇄ tools); data_analyst's
    │     first SQL query on "pricing" fails and is retried
    ├─ synthesize (fan-in of all teams)
    ├─ writer ◀──────── critic sends the first report back (loop)
    │     writer: outline ─▶ drafter ─▶ editor ─┐  (editor loops back to drafter once)
    │     drafter: section_writer (3 parallel tool calls) ─▶ fact_checker ─▶ compose
    └─ critic ─▶ publish ─▶ END

Four levels of agents deep (orchestrator → writer → drafter → section_writer)
and 17 agents in total. The LLMs are rule-based fakes that answer from the
conversation so far, so they're safe to run concurrently. Tools sleep a
little so the run is watchable live.

    python examples/deep_orchestration_demo.py --out agent-flow.jsonl
"""

import argparse
import operator
import threading
import time
import uuid
import zlib
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
# Python 3.9 resolves MessagesState's inherited hints in this module's namespace
from langchain_core.messages import AnyMessage  # noqa: F401
from langgraph.graph.message import add_messages  # noqa: F401
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from typing_extensions import TypedDict

from agent_flow_langgraph import AgentFlowCallbackHandler

DELAY = 0.6
TOPICS = ["market size", "churn drivers", "pricing"]
REQUEST = "Write a board-ready report on the SaaS analytics market: size, churn drivers and pricing"


def pause(key: str) -> None:
    """Sleep ~DELAY, jittered deterministically per call so parallel work staggers."""
    if DELAY > 0:
        time.sleep(DELAY * (0.6 + (zlib.crc32(key.encode()) % 9) / 10))


# ─── Rule-based fake chat model ──────────────────────────────────────────────

class RuleBasedChatModel(BaseChatModel):
    """Fake chat model whose reply is a pure function of the messages.

    Unlike iterator-scripted fakes it has no internal state, so one instance
    can serve many parallel runs of the same subgraph.
    """

    respond: Any  # Callable[[List[BaseMessage]], AIMessage]
    model: str = "demo-sonnet-4-5"

    @property
    def _llm_type(self) -> str:
        return "rule-based-fake"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        pause("llm:" + str(len(messages)))
        message = self.respond(messages)
        chars = sum(len(str(m.content)) for m in messages)
        message.usage_metadata = {
            "input_tokens": 1200 + chars // 3,
            "output_tokens": 80 + len(str(message.content)) // 3,
            "total_tokens": 1280 + chars // 3 + len(str(message.content)) // 3,
        }
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RuleBasedChatModel":
        return self


def _since_last_human(messages: Sequence[BaseMessage]) -> Tuple[str, List[BaseMessage]]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            return str(messages[i].content), list(messages[i + 1:])
    return "", list(messages)


def call(name: str, **args: Any) -> Dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:10]}"}


Turn = List[Tuple[str, Dict[str, Any]]]


def react_policy(
    tool_names: Sequence[str],
    plan: Callable[[str], List[Turn]],
    final: Callable[[str, List[ToolMessage]], str],
    thinking: Optional[Callable[[str], str]] = None,
) -> Callable[[List[BaseMessage]], AIMessage]:
    """Build a ReAct policy: run ``plan(task)`` turn by turn (a turn may hold
    several parallel tool calls), retry any call that errored, then answer
    with ``final``. Only this agent's own tool results count, so agents that
    share a message history don't confuse each other."""

    def respond(messages: List[BaseMessage]) -> AIMessage:
        task, since = _since_last_human(messages)
        mine = [m for m in since if isinstance(m, ToolMessage) and m.name in tool_names]
        last_ai = next((m for m in reversed(since) if isinstance(m, AIMessage) and m.tool_calls), None)
        if last_ai is not None:
            ids = {c["id"]: c for c in last_ai.tool_calls}
            failed = [ids[m.tool_call_id] for m in mine if m.tool_call_id in ids and m.status == "error"]
            if failed:
                return AIMessage(content="That failed, retrying.",
                                 tool_calls=[call(c["name"], **c["args"]) for c in failed])
        done = sum(1 for m in mine if m.status != "error")
        seen = 0
        for turn in plan(task):
            if done < seen + len(turn):
                content: Any = ""
                if thinking and seen == 0:
                    content = [{"type": "thinking", "thinking": thinking(task)}]
                return AIMessage(content=content, tool_calls=[call(n, **a) for n, a in turn])
            seen += len(turn)
        return AIMessage(content=final(task, [m for m in mine if m.status != "error"]))

    return respond


def topic_of(task: str) -> str:
    return next((t for t in TOPICS if t in task), task)


# ─── Tools ───────────────────────────────────────────────────────────────────

_sql_attempts: Dict[str, int] = {}
_sql_lock = threading.Lock()


@tool
def web_search(query: str) -> str:
    """Search the web."""
    pause("web:" + query)
    return f"4 sources on '{query}': analyst notes, vendor blogs, a Gartner summary, a HN thread"


@tool
def fetch_page(url: str) -> str:
    """Fetch and summarise a web page."""
    pause("fetch:" + url)
    return f"{url}: 2026 figures with YoY comparisons and a methodology note"


@tool
def search_papers(query: str) -> str:
    """Search academic papers."""
    pause("papers:" + query)
    return f"3 papers on '{query}' (2024-2026), 1 meta-analysis"


@tool
def read_paper(paper_id: str) -> str:
    """Read a paper's abstract and results."""
    pause("read:" + paper_id)
    return f"{paper_id}: effect sizes and a robustness check across 1.2k companies"


@tool
def run_sql(query: str) -> str:
    """Run a SQL query against the metrics warehouse."""
    pause("sql:" + query)
    with _sql_lock:
        attempt = _sql_attempts.get(query, 0)
        _sql_attempts[query] = attempt + 1
    if "pricing" in query and attempt == 0:
        raise TimeoutError("warehouse query timed out after 30s")
    return f"{query!r} -> 48 rows (cohorts by plan tier and quarter)"


@tool
def make_chart(spec: str) -> str:
    """Render a chart."""
    pause("chart:" + spec)
    return f"chart rendered: {spec}"


@tool
def write_section(section: str, content: str) -> str:
    """Write one report section."""
    pause("section:" + section + content[:20])
    return f"[{section}] {content}"


@tool
def verify_claim(claim: str) -> str:
    """Check a claim against the research findings."""
    pause("verify:" + claim)
    return f"verified: {claim} (2 supporting sources)"


@tool
def add_citations(style: str) -> str:
    """Attach citations to the draft."""
    pause("cite:" + style)
    return f"11 citations added ({style})"


# ─── Research team (runs 3× in parallel via Send) ────────────────────────────

class TeamState(MessagesState):
    topic: str
    findings: Annotated[list, operator.add]


class TeamOutput(TypedDict):
    findings: Annotated[list, operator.add]


def _specialist(name: str, tools: list, plan: Callable[[str], List[Turn]], summary: str):
    names = [t.name for t in tools]
    policy = react_policy(
        names, plan,
        final=lambda task, results: f"{name} on {topic_of(task)}: {summary} ({len(results)} tool results)",
        thinking=lambda task: f"I'll cover {topic_of(task)} with {', '.join(names)}.",
    )
    return create_react_agent(RuleBasedChatModel(respond=policy), tools, name=name)


def build_research_team():
    web = _specialist(
        "web_researcher", [web_search, fetch_page],
        lambda task: [
            [("web_search", {"query": f"{topic_of(task)} 2026"})],
            [("fetch_page", {"url": f"https://example.com/{topic_of(task).replace(' ', '-')}/a"}),
             ("fetch_page", {"url": f"https://example.com/{topic_of(task).replace(' ', '-')}/b"})],
        ],
        "market signals collected",
    )
    papers = _specialist(
        "paper_researcher", [search_papers, read_paper],
        lambda task: [
            [("search_papers", {"query": topic_of(task)})],
            [("read_paper", {"paper_id": f"arxiv:{zlib.crc32(topic_of(task).encode()) % 9000 + 1000}.0421"})],
        ],
        "peer-reviewed evidence summarised",
    )
    data = _specialist(
        "data_analyst", [run_sql, make_chart],
        lambda task: [
            [("run_sql", {"query": f"select * from metrics where topic = '{topic_of(task)}'"})],
            [("make_chart", {"spec": f"{topic_of(task)} by quarter"})],
        ],
        "warehouse numbers charted",
    )

    def lead(state: TeamState) -> dict:
        pause("lead:" + state["topic"])
        return {"messages": [HumanMessage(f"Research {state['topic']} for the board report")]}

    def merge(state: TeamState) -> dict:
        answers = [m.content for m in state["messages"] if isinstance(m, AIMessage) and not m.tool_calls and m.content]
        return {"findings": [f"{state['topic']}: " + " | ".join(answers)]}

    team = StateGraph(TeamState, output_schema=TeamOutput)
    team.add_node("lead", lead)
    team.add_node("web_researcher", web)
    team.add_node("paper_researcher", papers)
    team.add_node("data_analyst", data)
    team.add_node("merge", merge)
    team.add_edge(START, "lead")
    for specialist in ("web_researcher", "paper_researcher", "data_analyst"):
        team.add_edge("lead", specialist)
    team.add_edge(["web_researcher", "paper_researcher", "data_analyst"], "merge")
    team.add_edge("merge", END)
    return team.compile(name="research_team")


# ─── Writer (nested: writer → drafter → section_writer / fact_checker) ──────

SECTIONS = ["executive summary", "market & churn", "pricing & recommendations"]


class DraftState(MessagesState):
    report: str


def build_drafter():
    section_writer = create_react_agent(
        RuleBasedChatModel(respond=react_policy(
            ["write_section"],
            lambda task: [[("write_section", {"section": s, "content": f"{s} ({'revised' if 'Revise' in task else 'first pass'})"}) for s in SECTIONS]],
            final=lambda task, results: f"Drafted {len(results)} sections.",
            thinking=lambda task: "Three sections, written in parallel.",
        )),
        [write_section], name="section_writer",
    )
    fact_checker = create_react_agent(
        RuleBasedChatModel(respond=react_policy(
            ["verify_claim", "add_citations"],
            lambda task: [
                [("verify_claim", {"claim": "median churn is 3.5%/month"}),
                 ("verify_claim", {"claim": "usage-based pricing grew 2x since 2024"})],
                [("add_citations", {"style": "APA"})],
            ],
            final=lambda task, results: "All claims check out; citations attached.",
        )),
        [verify_claim, add_citations], name="fact_checker",
    )

    def compose(state: DraftState) -> dict:
        parts = [m.content for m in state["messages"] if isinstance(m, ToolMessage) and m.name == "write_section"]
        return {"report": "\n".join(parts[-len(SECTIONS):])}

    drafter = StateGraph(DraftState)
    drafter.add_node("section_writer", section_writer)
    drafter.add_node("fact_checker", fact_checker)
    drafter.add_node("compose", compose)
    drafter.add_edge(START, "section_writer")
    drafter.add_edge("section_writer", "fact_checker")
    drafter.add_edge("fact_checker", "compose")
    drafter.add_edge("compose", END)
    return drafter.compile(name="drafter")


class WriterState(MessagesState):
    findings: list
    report: str
    revisions: int
    edit_passes: int


class WriterOutput(TypedDict):
    report: str


def build_writer():
    def outline(state: WriterState) -> dict:
        pause("outline")
        note = " Address the critic's notes." if state.get("revisions") else ""
        return {"messages": [HumanMessage(f"Write the report from {len(state['findings'])} team findings.{note}")],
                "edit_passes": 0}

    def editor(state: WriterState) -> dict:
        pause("editor" + str(state.get("edit_passes", 0)))
        passes = state.get("edit_passes", 0) + 1
        update: dict = {"edit_passes": passes}
        if passes < 2:
            update["messages"] = [HumanMessage("Revise: tighten the executive summary")]
        return update

    writer = StateGraph(WriterState, output_schema=WriterOutput)
    writer.add_node("outline", outline)
    writer.add_node("drafter", build_drafter())
    writer.add_node("editor", editor)
    writer.add_edge(START, "outline")
    writer.add_edge("outline", "drafter")
    writer.add_edge("drafter", "editor")
    writer.add_conditional_edges("editor", lambda s: "drafter" if s["edit_passes"] < 2 else END, ["drafter", END])
    return writer.compile(name="writer")


# ─── Orchestrator ────────────────────────────────────────────────────────────

class OrchestratorState(MessagesState):
    topics: list
    findings: Annotated[list, operator.add]
    report: str
    revisions: int
    verdict: str


def _planner_reply(messages: List[BaseMessage]) -> AIMessage:
    return AIMessage(content=[
        {"type": "thinking", "thinking": "Three independent research threads, so fan out one team per topic, then synthesise and write."},
        {"type": "text", "text": "Plan: parallel research on " + ", ".join(TOPICS) + "; then synthesis, drafting and review."},
    ])


def _critic_reply(messages: List[BaseMessage]) -> AIMessage:
    task, _ = _since_last_human(messages)
    first = task.startswith("Review draft #1")
    return AIMessage(content="REVISE: pricing section needs numbers." if first else "APPROVE: ready for the board.")


planner_model = RuleBasedChatModel(respond=_planner_reply, model="demo-opus-4-1")
critic_model = RuleBasedChatModel(respond=_critic_reply, model="demo-opus-4-1")


def build_graph():
    def planner(state: OrchestratorState) -> dict:
        reply = planner_model.invoke(state["messages"])
        return {"topics": list(TOPICS), "messages": [reply]}

    def fan_out(state: OrchestratorState) -> list:
        return [Send("research_team", {"topic": t, "messages": []}) for t in state["topics"]]

    def synthesize(state: OrchestratorState) -> dict:
        pause("synthesize")
        return {"messages": [AIMessage(content=f"Synthesised {len(state['findings'])} team findings.")]}

    def critic(state: OrchestratorState) -> dict:
        draft = state.get("revisions", 0) + 1
        reply = critic_model.invoke(state["messages"] + [HumanMessage(f"Review draft #{draft}:\n" + state.get("report", ""))])
        return {"messages": [reply], "verdict": str(reply.content).split(":")[0],
                "revisions": state.get("revisions", 0) + 1}

    def publish(state: OrchestratorState) -> dict:
        pause("publish")
        return {"messages": [AIMessage(content="Published the report to the board portal.")]}

    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner)
    graph.add_node("research_team", build_research_team())
    graph.add_node("synthesize", synthesize)
    graph.add_node("writer", build_writer())
    graph.add_node("critic", critic)
    graph.add_node("publish", publish)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", fan_out, ["research_team"])
    graph.add_edge("research_team", "synthesize")
    graph.add_edge("synthesize", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", lambda s: "writer" if s["verdict"] == "REVISE" else "publish", ["writer", "publish"])
    graph.add_edge("publish", END)
    return graph.compile(name="orchestrator")


def run(out: str, delay: float = DELAY) -> dict:
    global DELAY
    DELAY = delay
    _sql_attempts.clear()
    graph = build_graph()
    handler = AgentFlowCallbackHandler(out, truncate=True, graph=graph)
    try:
        return graph.invoke(
            {"messages": [HumanMessage(REQUEST)], "revisions": 0},
            config={"callbacks": [handler], "recursion_limit": 50},
        )
    finally:
        handler.flush()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="agent-flow-events.jsonl", help="JSONL file to write")
    parser.add_argument("--delay", type=float, default=DELAY, help="base seconds per tool/LLM call")
    args = parser.parse_args(argv)
    result = run(args.out, args.delay)
    print(result["messages"][-1].content)
    print(f"events written to {args.out}")


if __name__ == "__main__":
    main()
