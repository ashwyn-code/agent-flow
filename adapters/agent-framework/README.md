# Agent Flow adapter for Microsoft Agent Framework

Streams [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) runs into the Agent Flow visualizer. It writes Agent Flow's JSONL event format, and Agent Flow replays that file and keeps following it as new events arrive.

It covers agents (streaming and non-streaming), agents as tools, and workflows, including switch-case routing, fan-out and fan-in, conditional loops and nested workflows.

## Install

```bash
pip install -e adapters/agent-framework
```

It needs Python 3.10 or later and `agent-framework-core` 1.0 or later.

## Use

```python
from agent_flow_agent_framework import AgentFlow

flow = AgentFlow("/tmp/agent-flow.jsonl", truncate=True)

agent = Agent(client, name="helper", tools=[...], middleware=flow.middleware)   # one agent
workflow = flow.instrument(builder.build())   # a workflow and every agent and sub-workflow in it
result = await workflow.run("Handle the alert")
```

Then open it in Agent Flow:

- **Standalone (no VS Code):** `npx agent-flow-app --event-log /tmp/agent-flow.jsonl`. From a clone of this repo, run `AGENT_FLOW_EVENT_LOG=/tmp/agent-flow.jsonl pnpm run dev`
- **VS Code:** set `agentVisualizer.eventLogPath` to `/tmp/agent-flow.jsonl`

`flow.instrument()` accepts an `Agent`, `AgentExecutor`, `WorkflowExecutor` or `Workflow`. It attaches to everything nested inside, including agents you exposed with `agent.as_tool()`. An agent that only exists inside your own tool function needs `middleware=flow.middleware` too. Adapter errors are caught and logged once. They never break your run.

To try it without an API key:

```bash
python adapters/agent-framework/examples/incident_response_demo.py --out /tmp/agent-flow.jsonl
```

The demo is an incident-response workflow with 12 agents, nested four levels deep:

- **Switch-case routing:** triage decides the severity. The low-severity branch exists but isn't taken.
- **Fan-out and fan-in:** three analyst agents run in parallel, and a correlator merges their findings.
- **Failure and retry:** one analyst's first metrics query fails, and it retries.
- **Agents as tools:** two subagents are exposed with `agent.as_tool()`.
- **Nested workflow:** the commander hands off to a remediation sub-workflow.
- **Conditional loop:** a failed verification sends the incident back to the commander.

`examples/fake_client.py` is a stateless rule-based chat client built from the framework's own layers, so function calling and middleware behave as they would with a real provider.

## How Agent Framework maps to Agent Flow

| Agent Framework | Agent Flow |
|---|---|
| First agent / workflow run with no caller | Main agent. Its input becomes the user message |
| Agent run inside another agent's tool (`as_tool()` or your own) | Subagent of that agent. Concurrent runs of the same agent get `name #2`, … |
| `Workflow` | An agent with a node graph: `graph_structure` (executors, start executor, edge groups; switch-case, multi-selection and conditional edges are dashed), plus `node_start` / `node_end` per execution |
| `AgentExecutor` / `WorkflowExecutor` node | Subagent of the workflow, named after the executor id. A nested workflow is a `subgraph` node that opens its own graph |
| Tool call (function middleware) | `tool_call_start` / `tool_call_end`, with `isError` when the tool raises |
| Model call (chat middleware) | `message` (text, plus `thinking` for reasoning), `model_detected`, and `context_update` from the call's token usage |

**How agents are observed:** through the framework's agent, function and chat middleware (`flow.middleware`). A context variable tracks which agent is running. For streaming runs, it is re-applied around every stream pull, so tool and model calls made while you consume the stream are attributed to the right agent.

**How workflows are observed:** workflows have no public observer API, so `instrument(workflow)` replaces two methods on that instance: the workflow's `run`, and each executor's `execute`. Classes are untouched. `execute` receives the ids of the executors whose messages triggered it, so every hop is exact, including fan-in from several sources. `__end__` is reached from the executors whose latest output nothing downstream used.

## Tests

```bash
pip install -e 'adapters/agent-framework[dev]'
pytest adapters/agent-framework/tests
```
