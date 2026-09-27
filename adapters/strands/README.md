# Agent Flow adapter for Strands Agents

Streams [Strands Agents](https://strandsagents.com) runs into the Agent Flow visualizer. It's a Strands `HookProvider` that writes Agent Flow's JSONL event format, and Agent Flow replays that file and keeps following it as new events arrive.

![The Strands demo in Agent Flow: three parallel branches, a failed tool call, a nested graph and a Swarm handoff loop](../../docs/media/strands.gif)

It covers single agents, agents as tools, `Graph` (including parallel batches, conditional edges, loops and nested graphs) and `Swarm`.

## Install

```bash
pip install -e adapters/strands
```

It needs Python 3.10 or later and `strands-agents` 1.10 or later.

## Use

```python
from agent_flow_strands import AgentFlowHooks

flow = AgentFlowHooks("/tmp/agent-flow.jsonl", truncate=True)

agent = Agent(tools=[...], hooks=[flow])   # a single agent
graph = flow.instrument(builder.build())   # or a Graph / Swarm and everything nested in it
graph("Plan the launch")
```

Then open it in Agent Flow:

- **Standalone (no VS Code):** `npx agent-flow-app --event-log /tmp/agent-flow.jsonl`. From a clone of this repo, run `AGENT_FLOW_EVENT_LOG=/tmp/agent-flow.jsonl pnpm run dev`
- **VS Code:** set `agentVisualizer.eventLogPath` to `/tmp/agent-flow.jsonl`

Share one `AgentFlowHooks` between all the agents you want in the same view. `instrument()` reaches every agent inside a Graph or Swarm, including nested ones. It can't reach an agent that only exists inside your own tool function (agents as tools), so give that one `hooks=[flow]` too. Hook errors are caught and logged once. They never break your agents.

To try it without an API key:

```bash
python adapters/strands/examples/orchestration_demo.py --out /tmp/agent-flow.jsonl
```

The demo has 19 agents, nested four levels deep, and uses every Strands multi-agent pattern:

- **Top-level Graph:** a planner fans out three branches that run in parallel, and a strategist merges them. A reviewer then either sends the plan back to the strategist (a conditional loop) or approves it and hands it to a publisher.
- **Nested Graph:** `market_research` fans out to two analysts and merges them. One analyst consults a statistician agent (an agent used as a tool).
- **Agents as tools:** `competitor_scan` runs three analyst agents at once, one per competitor. One pricing fetch fails and is retried.
- **Swarm:** `tech_review` hands work between architect, security and performance agents.

`examples/fake_model.py` is a stateless rule-based fake model, so it is safe to share across parallel agents.

## How Strands maps to Agent Flow

| Strands | Agent Flow |
|---|---|
| First agent / Graph / Swarm invoked with no caller | Main agent. Its prompt becomes the user message |
| Agent invoked inside another agent's tool | Subagent of that agent. The tool call is the dispatch, and the agent's answer is the return |
| `Graph` / `Swarm` | An agent with a node graph: `graph_structure` (nodes, entry points, conditional edges; Swarm handoffs are dynamic), plus `node_start` / `node_end` per node execution |
| Agent (or nested Graph) running in a node | Subagent of the graph, named after its node. A nested Graph is a `subgraph` node that opens its own graph |
| Tool call | `tool_call_start` / `tool_call_end`, with `isError` on exceptions or error results |
| Assistant message | `message`: text, plus `thinking` for reasoning content |
| Model | `model_detected` from the model config's `model_id`, and `context_update` from the latest context size |

Callers are found with context variables. Strands copies the context into the threads and tasks it runs agents, tools and nodes in, so a nested agent can see which tool or node started it. Names are unique among the agents running at the same moment: agents running concurrently get `name #2`, and so on, and a later agent can reuse a finished one's name.

**Graph routing:** a node's `from` lists the predecessors that produced new output since the node last ran, with each edge's condition re-checked. Edge conditions are therefore evaluated one extra time, so keep them free of side effects. `__end__` is reached from nodes whose latest output nothing downstream used. Swarm hops follow the handoff order.

## Sending events elsewhere

The constructor also takes `url=`, `token=`, `session=`, `content="metadata"`, `redact=` and `sample_rate=` (or the matching `AGENT_FLOW_*` environment variables). You can send events to a relay over HTTP, strip content, or sample runs. Events are delivered in the background, and `flush()` waits for them. See [Beyond your laptop](../../README.md#beyond-your-laptop-http-redaction-and-sampling) in the main README.

## Tests

```bash
pip install -e 'adapters/strands[dev]'
pytest adapters/strands/tests
```
