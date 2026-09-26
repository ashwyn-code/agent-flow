"""Multi-agent LangGraph demo for Agent Flow. Needs no API key.

A supervisor node hands work to two ReAct subagents (researcher, writer).
The LLMs are scripted fakes, and tools sleep briefly so the run is watchable
live.

    python examples/multi_agent_demo.py --out agent-flow.jsonl

Then point Agent Flow's ``agentVisualizer.eventLogPath`` setting at the file.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, List

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from agent_flow_langgraph import AgentFlowCallbackHandler

DELAY = 0.8


class ScriptedChatModel(GenericFakeChatModel):
    """Fake chat model that replays AIMessages and accepts bound tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self


def scripted(*messages: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))


def call(name: str, call_id: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": call_id}


@tool
def web_search(query: str) -> str:
    """Search the web."""
    time.sleep(DELAY)
    return f"3 articles about {query}: pricing tiers, churn benchmarks, competitor launches"


@tool
def read_url(url: str) -> str:
    """Fetch a page."""
    time.sleep(DELAY)
    return f"{url}: median SaaS churn is 3.5%/month; annual plans cut churn by ~40%"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file."""
    time.sleep(DELAY)
    return f"wrote {len(content)} chars to {path}"


def build_graph():
    researcher = create_react_agent(
        scripted(
            AIMessage(content="", tool_calls=[call("web_search", "r1", query="SaaS churn benchmarks 2026")]),
            AIMessage(content="", tool_calls=[call("read_url", "r2", url="https://example.com/churn-report")]),
            AIMessage(content="Median churn is 3.5%/month; annual plans reduce it by about 40%."),
        ),
        [web_search, read_url],
        name="researcher",
    )
    writer = create_react_agent(
        scripted(
            AIMessage(content="", tool_calls=[call("write_file", "w1", path="report.md", content="# Churn report\n...")]),
            AIMessage(content="Report written to report.md."),
        ),
        [write_file],
        name="writer",
    )

    def supervisor(state: MessagesState) -> dict:
        time.sleep(DELAY)
        return {"messages": [AIMessage(content="Plan: research churn data, then write the report.")]}

    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)
    return graph.compile(name="supervisor")


def main(argv: List[str] = None) -> None:
    global DELAY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="agent-flow-events.jsonl", help="JSONL file to write")
    parser.add_argument("--delay", type=float, default=DELAY, help="seconds each tool/node sleeps")
    args = parser.parse_args(argv)
    DELAY = args.delay

    graph = build_graph()
    handler = AgentFlowCallbackHandler(args.out, truncate=True, graph=graph)
    result = graph.invoke(
        {"messages": [HumanMessage("Write a short report on SaaS churn benchmarks")]},
        config={"callbacks": [handler]},
    )
    print(result["messages"][-1].content)
    print(f"events written to {args.out}")


if __name__ == "__main__":
    main()
