# Agent Flow adapter for CrewAI

Streams [CrewAI](https://www.crewai.com) crews and flows into the Agent Flow visualizer. It listens on CrewAI's event bus and writes Agent Flow's JSONL event format, and Agent Flow replays that file and keeps following it as new events arrive.

![The launch Flow demo in Agent Flow: parallel branches, a router loop, and drilling into the crews the Flow ran](../../docs/media/crewai.gif)

It covers crews (sequential, parallel async tasks, task `context`, hierarchical managers and delegation), flows (`@start`, `@listen` with `and_` / `or_`, and `@router` loops), crews and agents run from Flow methods, and direct `agent.kickoff()` calls.

## Install

```bash
pip install -e adapters/crewai
```

It needs Python 3.10–3.13, the range CrewAI supports, and `crewai` 1.0 or later.

## Use

```python
from agent_flow_crewai import AgentFlowListener

listener = AgentFlowListener("/tmp/agent-flow.jsonl", truncate=True)
MyFlow().kickoff()        # or crew.kickoff(), agent.kickoff(...)
listener.close()          # flush and stop listening (also runs at exit)
```

Then open it in Agent Flow:

- **Standalone (no VS Code):** `npx agent-flow-app --event-log /tmp/agent-flow.jsonl`. From a clone of this repo, run `AGENT_FLOW_EVENT_LOG=/tmp/agent-flow.jsonl pnpm run dev`
- **VS Code:** set `agentVisualizer.eventLogPath` to `/tmp/agent-flow.jsonl`

You don't have to attach anything to your agents. The listener sees everything that runs in the process while it is open. Adapter errors are caught and logged once. They never break your crew.

To try it without an API key:

```bash
python adapters/crewai/examples/launch_flow_demo.py --out /tmp/agent-flow.jsonl
```

The demo is a product-launch Flow with 14 agents, nested four levels deep:

- **Flow fan-out:** two branches run in parallel, a research crew and a creative crew.
- **Parallel async tasks:** in the research crew, two async tasks run concurrently and merge through `context`. One pricing lookup fails, and the agent retries it.
- **Hierarchical crew:** the Crew Manager delegates work to a copywriter and asks an editor a question.
- **Router loop:** a router sends the draft back for one polish pass, then approves it.
- **Direct agent runs:** the polish step and the publisher are single agents started straight from Flow methods.

`examples/fake_llm.py` is a rule-based fake `BaseLLM` that speaks CrewAI's ReAct format and emits its LLM events.

## How CrewAI maps to Agent Flow

| CrewAI | Agent Flow |
|---|---|
| First crew / flow / agent run with no caller | Main agent. A crew's first task or its inputs become the user message |
| `Crew` | An agent with a node graph of its tasks: order, async tasks (parallel) and `context` dependencies are edges, and each task execution is a node |
| `Flow` | An agent with a node graph of its methods: `@listen` edges (including `and_` / `or_`) and `@router` label edges (dashed, labelled), and each method execution is a node |
| Agent executing a task | Subagent of the crew, named after its role |
| Crew or agent run inside a Flow method | Subagent of the flow |
| Coworker run from "Delegate work to coworker" / "Ask question to coworker" | Subagent of the delegating agent |
| Tool usage | `tool_call_start` / `tool_call_end`, one per attempt, with `isError` on errors |
| LLM call | `message` (the ReAct thought as `thinking`, the final answer), `model_detected`, and `context_update` from token usage |

**How parents are found:** CrewAI attaches a `parent_event_id` scope chain to every event. The listener walks it to find the task, method or tool call an agent was started from.

**Ordering:** CrewAI's event bus runs handlers on a thread pool, so events can arrive out of order. The listener holds each event for a quarter of a second and writes events in emit order, which adds that much latency to the live view.

**Routing:** a node's `from` lists the predecessors that produced new output since it last ran. For router edges, the router's returned label has to match. A router's labels appear in the static graph only when declared, as `@router(..., emit=[...])` or with a `Literal` return type. Otherwise the taken route still shows up as it runs.

**Known CrewAI behaviors you'll see in the trace:**
- **Crew kickoffs are serialized.** Crews started from parallel Flow methods run one after another, even though the methods start together.
- **Silent tool retries.** CrewAI sometimes retries a failing tool internally before any event fires. Tool errors that CrewAI reports show up as failed calls.

## Sending events elsewhere

The constructor also takes `url=`, `token=`, `session=`, `content="metadata"`, `redact=` and `sample_rate=` (or the matching `AGENT_FLOW_*` environment variables). You can send events to a relay over HTTP, strip content, or sample runs. Events are delivered in the background, and `flush()` waits for them. See [Beyond your laptop](../../README.md#beyond-your-laptop-http-redaction-and-sampling) in the main README.

## Tests

```bash
pip install -e 'adapters/crewai[dev]'
pytest adapters/crewai/tests
```

The tests keep CrewAI offline and point `HOME` at a temporary directory. CrewAI writes to your home directory otherwise.
