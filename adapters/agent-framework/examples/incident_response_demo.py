"""Nested, fan-out Microsoft Agent Framework demo for Agent Flow. No API key.

    incident_response (Workflow)
    ├─ intake ─▶ triage (agent)
    ├─ switch-case on severity:
    │    high ─▶ dispatch ──fan-out──▶ log_analyst     ┐
    │                              ─▶ metrics_analyst  ├──fan-in──▶ correlate
    │                              ─▶ deploy_auditor   ┘
    │           (parallel agents; metrics_analyst's first query fails and is
    │            retried; deploy_auditor asks git_historian via agent.as_tool())
    │    default ─▶ auto_ack   (not taken)
    ├─ commander (agent) ─▶ remediation (nested Workflow)
    │     remediation: planner ─▶ operator ─▶ verifier
    │     operator delegates to k8s_bot via agent.as_tool()
    ├─ remediation ──FAIL──▶ commander (loop, once)  /  ──PASS──▶ postmortem
    └─ postmortem (agent) ─▶ publish

12 agents, four levels deep, using switch-case routing, fan-out and
fan-in, a conditional loop, a nested workflow and agents as tools.

    python examples/incident_response_demo.py --out agent-flow.jsonl
"""

import argparse
import asyncio
import logging
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_framework import (
    Agent,
    AgentExecutor,
    AgentExecutorResponse,
    Case,
    Default,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowExecutor,
    executor,
    tool,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_client import RuleBasedChatClient, react  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_flow_agent_framework import AgentFlow  # noqa: E402

DELAY = 0.5
ALERT = "SEV1: checkout API error rate at 38% since 14:02 UTC, p99 latency 9s"


def pause(key: str) -> None:
    if DELAY > 0:
        time.sleep(DELAY * (0.6 + (zlib.crc32(key.encode()) % 9) / 10))


def client(policy: Any, model: str = "demo-gpt-5-mini") -> RuleBasedChatClient:
    return RuleBasedChatClient(policy, model=model, delay=lambda: DELAY * 0.4)


def say(text: str, thinking: Optional[str] = None) -> Any:
    def policy(messages: Any) -> Dict[str, Any]:
        return {"text": text, **({"thinking": thinking} if thinking else {})}
    return policy


# ─── Tools ───────────────────────────────────────────────────────────────────

_attempts: Dict[str, int] = {}
_lock = threading.Lock()


@tool
def search_logs(query: str) -> str:
    """Search production logs."""
    pause("logs:" + query)
    return f"'{query}': 41k matches, first at 14:02:11 in checkout-api pods"


@tool
def query_metrics(metric: str) -> str:
    """Query the metrics store."""
    pause("metrics:" + metric)
    with _lock:
        attempt = _attempts.get(metric, 0)
        _attempts[metric] = attempt + 1
    if metric == "db_pool_saturation" and attempt == 0:
        raise TimeoutError("metrics store timed out")
    return f"{metric}: spiked to 100% at 14:02, flat before"


@tool
def list_deploys(service: str) -> str:
    """List recent deploys."""
    pause("deploys:" + service)
    return f"{service}: v2.41.0 deployed 14:00 UTC by ci-bot"


@tool
def git_log(ref: str) -> str:
    """Show commits in a release."""
    pause("git:" + ref)
    return f"{ref}: 'lower db pool size to 8' (a1b2c3), 'bump client' (d4e5f6)"


@tool
def kubectl(command: str) -> str:
    """Run a kubectl command."""
    pause("kubectl:" + command)
    return f"$ kubectl {command}\nok"


@tool
def post_status(channel: str) -> str:
    """Post to the status page."""
    pause("status:" + channel)
    return f"posted to {channel}"


# ─── Agents ──────────────────────────────────────────────────────────────────

def build(flow: AgentFlow):
    git_historian = Agent(client(react(
        lambda task: [[("git_log", {"ref": "v2.41.0"})]],
        lambda task, results: "v2.41.0 lowered the DB pool from 32 to 8 connections.",
    )), name="git_historian", tools=[git_log])

    k8s_bot = Agent(client(react(
        lambda task: [[("kubectl", {"command": "rollout undo deploy/checkout-api"})],
                      [("kubectl", {"command": "rollout status deploy/checkout-api"})]],
        lambda task, results: "Rolled back checkout-api to v2.40.3; rollout healthy.",
        thinking=lambda task: "Roll back, then wait for the rollout.",
    )), name="k8s_bot", tools=[kubectl])

    triage = Agent(client(say("Severity: HIGH. Customer-facing checkout outage.",
                              thinking="Error rate and latency both breach SEV1 thresholds."),
                          model="demo-gpt-5"), name="triage")
    log_analyst = Agent(client(react(
        lambda task: [[("search_logs", {"query": "checkout 5xx"}), ("search_logs", {"query": "connection pool timeout"})]],
        lambda task, results: "Logs: pool timeouts started with the 14:00 deploy.",
    )), name="log_analyst", tools=[search_logs])
    metrics_analyst = Agent(client(react(
        lambda task: [[("query_metrics", {"metric": "db_pool_saturation"})], [("query_metrics", {"metric": "error_rate"})]],
        lambda task, results: "Metrics: DB pool saturated at 14:02, error rate followed.",
    )), name="metrics_analyst", tools=[query_metrics])
    deploy_auditor = Agent(client(react(
        lambda task: [[("list_deploys", {"service": "checkout-api"})], [("git_historian", {"task": "What changed in v2.41.0?"})]],
        lambda task, results: "Deploys: v2.41.0 at 14:00 shrank the DB pool.",
        thinking=lambda task: "Check what shipped right before the incident.",
    )), name="deploy_auditor", tools=[list_deploys, git_historian.as_tool(description="Ask the git historian about a release")])

    decisions: List[str] = []

    def commander_policy(messages: Any) -> Dict[str, Any]:
        text = ("Remediate: roll back v2.41.0." if not decisions else
                "Rollback verified failing; restore the pool size config and retry.")
        decisions.append(text)
        return {"text": text, "thinking": "Weigh rollback against a forward fix."}

    commander = Agent(client(commander_policy, model="demo-gpt-5"), name="commander")
    planner = Agent(client(say("Plan: 1) roll back checkout-api 2) confirm error rate < 1%.")), name="planner")
    operator = Agent(client(react(
        lambda task: [[("k8s_bot", {"task": "Roll back checkout-api"})]],
        lambda task, results: "Rollback executed via k8s_bot.",
    )), name="operator", tools=[k8s_bot.as_tool(description="Delegate a Kubernetes change")])
    postmortem = Agent(client(react(
        lambda task: [[("post_status", {"channel": "status-page"}), ("post_status", {"channel": "#incidents"})]],
        lambda task, results: "Postmortem drafted: pool-size regression in v2.41.0, 38 minutes of impact.",
    ), model="demo-gpt-5"), name="postmortem", tools=[post_status])

    # ─── Nested remediation workflow ──────────────────────────────────────────
    verifications: List[str] = []

    @executor(id="verifier")
    async def verifier(response: AgentExecutorResponse, ctx: WorkflowContext[str, str]) -> None:
        await asyncio.to_thread(pause, "verify")
        verdict = "FAIL: error rate still 12%" if not verifications else "PASS: error rate 0.3%"
        verifications.append(verdict)
        await ctx.yield_output(verdict)

    planner_exec = AgentExecutor(planner, id="planner")
    operator_exec = AgentExecutor(operator, id="operator")
    remediation_flow = (WorkflowBuilder(start_executor=planner_exec, name="remediation")
                        .add_edge(planner_exec, operator_exec)
                        .add_edge(operator_exec, verifier)
                        .build())

    # ─── Top-level workflow ───────────────────────────────────────────────────
    @executor(id="intake")
    async def intake(alert: str, ctx: WorkflowContext[str]) -> None:
        await asyncio.to_thread(pause, "intake")
        await ctx.send_message(f"Triage this alert: {alert}")

    @executor(id="dispatch")
    async def dispatch(response: AgentExecutorResponse, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(f"Investigate: {ALERT}. Triage said: {response.agent_response.text}")

    @executor(id="auto_ack")
    async def auto_ack(response: AgentExecutorResponse, ctx: WorkflowContext[str, str]) -> None:
        await ctx.yield_output("Low severity: acknowledged automatically.")

    @executor(id="correlate")
    async def correlate(responses: list[AgentExecutorResponse], ctx: WorkflowContext[str]) -> None:
        await asyncio.to_thread(pause, "correlate")
        findings = "\n".join(f"- {r.executor_id}: {r.agent_response.text}" for r in responses)
        await ctx.send_message(f"Findings from {len(responses)} analysts:\n{findings}")

    @executor(id="publish")
    async def publish(response: AgentExecutorResponse, ctx: WorkflowContext[str, str]) -> None:
        await ctx.yield_output(f"Incident closed. {response.agent_response.text}")

    triage_exec = AgentExecutor(triage, id="triage")
    analysts = [AgentExecutor(log_analyst, id="log_analyst"), AgentExecutor(metrics_analyst, id="metrics_analyst"),
                AgentExecutor(deploy_auditor, id="deploy_auditor")]
    commander_exec = AgentExecutor(commander, id="commander")
    remediation = WorkflowExecutor(remediation_flow, id="remediation")
    postmortem_exec = AgentExecutor(postmortem, id="postmortem")

    def is_high(response: Any) -> bool:
        return "HIGH" in response.agent_response.text

    workflow = (WorkflowBuilder(start_executor=intake, name="incident_response", max_iterations=40,
                                output_from=[publish, auto_ack])
                .add_edge(intake, triage_exec)
                .add_switch_case_edge_group(triage_exec, [Case(condition=is_high, target=dispatch), Default(target=auto_ack)])
                .add_fan_out_edges(dispatch, analysts)
                .add_fan_in_edges(analysts, correlate)
                .add_edge(correlate, commander_exec)
                .add_edge(commander_exec, remediation)
                .add_edge(remediation, commander_exec, condition=lambda verdict: str(verdict).startswith("FAIL"))
                .add_edge(remediation, postmortem_exec, condition=lambda verdict: str(verdict).startswith("PASS"))
                .add_edge(postmortem_exec, publish)
                .build())
    return flow.instrument(workflow)


async def run(out: str, delay: float = DELAY) -> Any:
    global DELAY
    DELAY = delay
    _attempts.clear()
    flow = AgentFlow(out, truncate=True)
    workflow = build(flow)
    try:
        return await workflow.run(ALERT)
    finally:
        flow.flush()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="agent-flow-events.jsonl", help="JSONL file to write")
    parser.add_argument("--delay", type=float, default=DELAY, help="base seconds per tool / model call")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)  # the framework logs the demo's deliberate tool failure
    result = asyncio.run(run(args.out, args.delay))
    print("outputs:", result.get_outputs())
    print(f"events written to {args.out}")


if __name__ == "__main__":
    main()
