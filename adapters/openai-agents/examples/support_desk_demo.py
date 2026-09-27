"""Nested, fan-out OpenAI Agents SDK demo for Agent Flow. No API key, no network.

    trace("Support desk")
    ├─ in parallel:
    │   ├─ Sentiment Analyst (scores the ticket's tone)
    │   └─ Triage (input guardrail: pii_filter)
    │        ──handoff──▶ Tech Support    (Billing and Sales are declared too)
    │           one turn, three parallel tool calls:
    │             search_kb, check_service_status (fails once, retried),
    │             analyze_logs = Log Analyst.as_tool()
    │                 grep_logs + inspect_db = DB Inspector.as_tool()  ── run_query
    │        ──handoff──▶ Billing (issue_credit; output guardrail: refund_policy)

Handoffs as a routing graph (with an unused route), parallel runs in one
trace, parallel tool calls, agents as tools four levels deep, a failing tool
that is retried, and input/output guardrails.

    python examples/support_desk_demo.py --out agent-flow.jsonl
"""

import argparse
import asyncio
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import (
    Agent,
    GuardrailFunctionOutput,
    Runner,
    function_tool,
    input_guardrail,
    output_guardrail,
    trace,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_model import RuleBasedModel, react  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_flow_openai_agents import install  # noqa: E402

DELAY = 0.5
TICKET = "Our dashboards have been timing out since this morning and we lost a day of reporting. Can you fix it and credit us?"


async def pause(key: str) -> None:
    if DELAY > 0:
        await asyncio.sleep(DELAY * (0.6 + (zlib.crc32(key.encode()) % 9) / 10))


def model(policy: Any, name: str = "demo-gpt-5-mini") -> RuleBasedModel:
    return RuleBasedModel(policy, name=name, delay=lambda: DELAY * 0.4)


def say(text: str, thinking: Optional[str] = None) -> Any:
    return lambda items: {"text": text, **({"thinking": thinking} if thinking else {})}


# ─── Tools ───────────────────────────────────────────────────────────────────

_status_calls: List[int] = []


@function_tool
async def analyze_tone(text: str) -> str:
    """Score the tone of a support ticket."""
    await pause("tone")
    return "frustrated (0.82), churn risk: medium"


@function_tool
async def search_kb(query: str) -> str:
    """Search the knowledge base."""
    await pause("kb:" + query)
    return "KB-2291: dashboard timeouts after the v8.3 query planner rollout; workaround: disable planner v2"


@function_tool
async def check_service_status(service: str) -> str:
    """Check a service's health."""
    await pause("status:" + service)
    _status_calls.append(1)
    if len(_status_calls) == 1:
        raise TimeoutError("status API did not respond")
    return f"{service}: degraded since 08:12 UTC (p95 latency 14s)"


@function_tool
async def grep_logs(pattern: str) -> str:
    """Search production logs."""
    await pause("grep:" + pattern)
    return f"'{pattern}': 3,412 hits, all on reporting-db-2"


@function_tool
async def run_query(sql: str) -> str:
    """Run a read-only SQL query."""
    await pause("sql:" + sql)
    return "planner_v2 enabled on reporting-db-2 since 08:10; 94% of slow queries use a seq scan"


@function_tool
async def issue_credit(account: str, days: int) -> str:
    """Issue a service credit."""
    await pause("credit")
    return f"credited {days} day(s) to {account}"


@input_guardrail
async def pii_filter(ctx: Any, agent: Any, input: Any) -> GuardrailFunctionOutput:
    await pause("pii")
    return GuardrailFunctionOutput(output_info={"pii": False}, tripwire_triggered=False)


@output_guardrail
async def refund_policy(ctx: Any, agent: Any, output: Any) -> GuardrailFunctionOutput:
    await pause("policy")
    return GuardrailFunctionOutput(output_info={"within_policy": True}, tripwire_triggered=False)


# ─── Agents ──────────────────────────────────────────────────────────────────

def build() -> Dict[str, Agent]:
    db_inspector = Agent(name="DB Inspector", instructions="Inspect the database.", tools=[run_query], model=model(react(
        lambda task: [[("run_query", {"sql": "select planner, count(*) from slow_queries group by 1"})]],
        lambda task, results: "planner_v2 on reporting-db-2 is behind the slow queries.",
    )))
    log_analyst = Agent(name="Log Analyst", instructions="Find the cause in the logs.", model=model(react(
        lambda task: [[("grep_logs", {"pattern": "statement timeout"}),
                       ("inspect_db", {"input": "Which planner are the slow queries using?"})]],
        lambda task, results: "Timeouts all come from reporting-db-2, where planner_v2 was enabled at 08:10.",
        thinking=lambda task: "Logs and the database in parallel.",
    )), tools=[grep_logs, db_inspector.as_tool(tool_name="inspect_db", tool_description="Ask the DB inspector")])

    billing = Agent(name="Billing", instructions="Handle credits.", tools=[issue_credit], output_guardrails=[refund_policy],
                    model=model(react(lambda task: [[("issue_credit", {"account": "acme-analytics", "days": 1})]],
                                      lambda task, results: "Fixed and credited: planner_v2 was rolled back and 1 day was credited.")))
    sales = Agent(name="Sales", instructions="Handle upgrades.", model=model(say("Happy to talk about plans.")))
    tech_support = Agent(name="Tech Support", instructions="Diagnose and fix.", handoffs=[billing], model=model(react(
        lambda task: [[("search_kb", {"query": "dashboard timeouts"}),
                       ("check_service_status", {"service": "reporting"}),
                       ("analyze_logs", {"input": "Why are dashboard queries timing out?"})],
                      [("transfer_to_billing", {})]],
        lambda task, results: "Resolved.",
        thinking=lambda task: "Check the KB, service health and logs at once.",
    ), name="demo-gpt-5"), tools=[
        search_kb, check_service_status,
        log_analyst.as_tool(tool_name="analyze_logs", tool_description="Ask the log analyst"),
    ])
    triage = Agent(name="Triage", instructions="Route the ticket.", handoffs=[billing, tech_support, sales],
                   input_guardrails=[pii_filter], model=model(react(
                       lambda task: [[("transfer_to_tech_support", {})]],
                       lambda task, results: "Routed.",
                       thinking=lambda task: "Outage first, then the credit: Tech Support.",
                   ), name="demo-gpt-5"))
    sentiment = Agent(name="Sentiment Analyst", instructions="Score tone.", tools=[analyze_tone], model=model(react(
        lambda task: [[("analyze_tone", {"text": task})]],
        lambda task, results: "Frustrated customer, medium churn risk: prioritise.",
    )))
    return {"triage": triage, "sentiment": sentiment}


async def run(out: str, delay: float = DELAY) -> Any:
    global DELAY
    DELAY = delay
    _status_calls.clear()
    install(out, truncate=True, exclusive=True)  # exclusive: no upload to the OpenAI trace dashboard
    agents = build()
    with trace("Support desk"):
        support, mood = await asyncio.gather(Runner.run(agents["triage"], TICKET), Runner.run(agents["sentiment"], TICKET))
    return support.final_output, mood.final_output


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="agent-flow-events.jsonl", help="JSONL file to write")
    parser.add_argument("--delay", type=float, default=DELAY, help="base seconds per tool / model call")
    args = parser.parse_args(argv)
    support, mood = asyncio.run(run(args.out, args.delay))
    print("support:", support)
    print("sentiment:", mood)
    print(f"events written to {args.out}")


if __name__ == "__main__":
    main()
