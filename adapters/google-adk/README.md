# Agent Flow adapter for Google ADK

Streams [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) runs into the Agent Flow visualizer. It's an ADK plugin that writes Agent Flow's JSONL event format, and Agent Flow replays that file and keeps following it as new events arrive.

![The trip planner demo in Agent Flow: a sequential pipeline with a parallel stage and a revision loop](../../docs/media/google-adk.gif)

It covers `LlmAgent` with tools and sub-agents (including `transfer_to_agent`), `SequentialAgent`, `ParallelAgent` and `LoopAgent` (nested in any combination), agents used as tools (`AgentTool`), parallel tool calls, and streaming.

## Install

```bash
pip install -e adapters/google-adk
```

It needs Python 3.10 or later and `google-adk` 1.5 or later.

## Use

```python
from agent_flow_adk import AgentFlowPlugin

plugin = AgentFlowPlugin("/tmp/agent-flow.jsonl", truncate=True)
app = App(name="trip_planner", root_agent=root_agent, plugins=[plugin])
runner = Runner(app=app, session_service=InMemorySessionService())
```

With older ADK versions, pass `plugins=[plugin]` to `Runner` or `InMemoryRunner` directly.

Then open it in Agent Flow:

- **Standalone (no VS Code):** `npx agent-flow-app --event-log /tmp/agent-flow.jsonl`. From a clone of this repo, run `AGENT_FLOW_EVENT_LOG=/tmp/agent-flow.jsonl pnpm run dev`
- **VS Code:** set `agentVisualizer.eventLogPath` to `/tmp/agent-flow.jsonl`

One plugin sees the whole run. ADK hands the runner's plugins to the nested runner that each `AgentTool` starts, so agents used as tools are included automatically. Every callback returns `None`, so the plugin never changes what your agents do, and it swallows its own errors. That matters because ADK aborts a run when a plugin raises.

To try it without an API key:

```bash
python adapters/google-adk/examples/trip_planner_demo.py --out /tmp/agent-flow.jsonl
```

The demo is a trip planner with 12 agents, nested four levels deep. The root is a `SequentialAgent` with four stages:

- **Intake:** an `LlmAgent` loads the traveller's profile.
- **Research:** a `ParallelAgent` runs flights, hotels and weather agents at once. The fare search fails once and is retried, and the hotel agent makes two tool calls in parallel.
- **Itinerary:** a `LoopAgent` alternates a planner and a critic. The planner asks a budget analyst through `AgentTool`. The critic sends the plan back once, then ends the loop with `exit_loop`.
- **Booking:** a booking agent hands over to its `reservations` sub-agent with `transfer_to_agent`.

`examples/fake_llm.py` is a stateless rule-based `BaseLlm` that reports model names, thoughts and token usage.

## How ADK maps to Agent Flow

| ADK | Agent Flow |
|---|---|
| The runner's root agent | Main agent. The user's message becomes its prompt |
| Sub-agent, or transfer target (`transfer_to_agent`) | Subagent of its parent agent |
| `SequentialAgent` | An agent with a node graph: its children in a chain |
| `ParallelAgent` | An agent with a node graph: its children fanned out from START, running concurrently |
| `LoopAgent` | A chain with a dashed loop-back edge. Each iteration re-runs the nodes, so they show `×N` |
| Nested workflow agent | A subgraph node that opens its own graph |
| `AgentTool` | The wrapped agent is a subagent of the agent that called the tool. Concurrent calls get `name #2`, … |
| Tool call | `tool_call_start` / `tool_call_end`. `isError` is set when the tool raises or returns `{"status": "error", ...}` (ADK's convention) |
| Model call | `message` (text, plus `thinking` for thought parts), `model_detected`, and `context_update` from `usage_metadata` |

A few details:

- **Linking `AgentTool` runs:** ADK runs `AgentTool` in a nested runner with a new invocation id and no link to the caller. The plugin carries the link in a context variable that it sets for the duration of the tool call.
- **Transfers:** ADK doesn't signal when a coordinator that transferred control finishes, so the plugin closes it when the run ends.
- **Name-based tracking:** agents are tracked by name, which is unique within an ADK agent tree. ADK 2.x's node runtime runs copies of your agent objects, so object identity can't be used.

**Not yet covered:** ADK 2.x's `Workflow` graph API (`google.adk.workflow`). `LlmAgent` nodes inside a `Workflow` show up as agents, but function and join nodes report only their completion and are not drawn.

## Sending events elsewhere

The constructor also takes `url=`, `token=`, `session=`, `content="metadata"`, `redact=` and `sample_rate=` (or the matching `AGENT_FLOW_*` environment variables). You can send events to a relay over HTTP, strip content, or sample runs. Events are delivered in the background, and `flush()` waits for them. See [Beyond your laptop](../../README.md#beyond-your-laptop-http-redaction-and-sampling) in the main README.

## Tests

```bash
pip install -e 'adapters/google-adk[dev]'
pytest adapters/google-adk/tests
```
