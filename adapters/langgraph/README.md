# Agent Flow adapter for LangGraph

Streams LangGraph runs into the Agent Flow visualizer. It's a LangChain callback handler that writes Agent Flow's JSONL event format. Agent Flow replays that file and keeps following it as new events arrive, in the standalone web app or in VS Code.

## Install

```bash
pip install -e adapters/langgraph
```

It needs Python 3.9 or later and `langchain-core` 0.3 or later. You install LangGraph yourself.

## Use

```python
from agent_flow_langgraph import AgentFlowCallbackHandler

handler = AgentFlowCallbackHandler("/tmp/agent-flow.jsonl", truncate=True, graph=graph)
graph.invoke(inputs, config={"callbacks": [handler]})   # or ainvoke / stream / astream
```

Then open it in Agent Flow, either way:

- **Standalone (no VS Code):** `npx agent-flow-app --event-log /tmp/agent-flow.jsonl`. From a clone of this repo, run `AGENT_FLOW_EVENT_LOG=/tmp/agent-flow.jsonl pnpm run dev` and open http://localhost:3000
- **VS Code:** set `agentVisualizer.eventLogPath` to `/tmp/agent-flow.jsonl` and run **Agent Flow: Open Agent Flow**

Agent Flow replays what's already in the file and then follows new lines, so it works during a live run and after the run finishes. In the standalone app the log gets its own session tab, labeled with the prompt, and each `truncate=True` run opens a fresh one.

Tip: if you set `AGENT_FLOW_EVENT_LOG` once in your shell, both the adapter (when you don't pass `path`) and Agent Flow use it.

To try it without an API key, run one of the demos. Both use fake LLMs.

```bash
# Small: a supervisor hands off to two ReAct agents
python adapters/langgraph/examples/multi_agent_demo.py --out /tmp/agent-flow.jsonl

# Large: nested multi-agent orchestration, about 20 s at the default --delay 0.6
python adapters/langgraph/examples/deep_orchestration_demo.py --out /tmp/agent-flow.jsonl
```

The large demo (`deep_orchestration_demo.py`) has 17 agents, nested four levels deep:

- **Parallel teams:** a planner sends 3 research teams out at once with `Send`. In each team, a lead fans out to three specialist agents that run in parallel, and a merge node joins their results.
- **Failure and retry:** one SQL tool call fails, and the agent retries it.
- **Nested writer:** the writer subgraph contains a drafter subgraph, which runs a section writer (three parallel tool calls) and then a fact checker.
- **Two review loops:** an editor sends the draft back to the drafter once, and a critic sends the report back to the writer once.

`tests/test_deep_orchestration.py` runs this demo as an end-to-end test.

### Options

| Argument | Default | |
|---|---|---|
| `path` | `$AGENT_FLOW_EVENT_LOG`, then `./agent-flow-events.jsonl` | File to append events to |
| `main_agent_name` | the top-level graph's `name` | Label for the main agent |
| `truncate` | `False` | Empty the file before the first event |
| `graph` | `None` | The compiled graph. Sends its declared structure, including every subgraph, so the Graph panel can also draw routes that weren't taken |

One handler instance is one session. Reusing it across `invoke` calls adds each run to the same view.

## How LangGraph maps to Agent Flow

| LangGraph | Agent Flow |
|---|---|
| Top-level graph run | Main agent. The input's last human message becomes the user prompt |
| Subgraph running in a node (including `create_react_agent`, and nested graphs at any depth) | Subagent named after the node. You get `subagent_dispatch` and `agent_spawn` when it starts, `subagent_return` (its final message) and `agent_complete` when it ends |
| Parallel runs of the same subgraph (`Send` fan-out) | Separate subagents: `worker`, `worker #2`, … |
| Tool call | `tool_call_start` / `tool_call_end`, with `isError` set when the tool raises |
| LLM response | `message` (text, plus `thinking` for reasoning blocks from Anthropic or OpenAI), `context_update` from `usage_metadata`, `model_detected` from `ls_model_name` |

| Node execution | `node_start` (with the superstep and the nodes it was reached from) and `node_end` (with `error` if it raised) |
| `graph=` structure | `graph_structure`: nodes plus normal and conditional edges, one per graph |

Plain function nodes aren't shown as agents. LLM and tool calls made inside them count toward the graph that contains them. The adapter identifies agents from the `langgraph_checkpoint_ns` metadata that LangGraph attaches to every callback.

## Graph panel

Press **Graph** in the top bar (or `N`) to see the run as the graph it actually is, next to the agent tree:

- **Layout:** nodes run top to bottom from START to END. Loops (like ReAct's agent → tools → agent) are arcs on the right, and parallel branches sit side by side above the node where they merge.
- **Routes:** routes taken are bright, and routes not taken are dim. Dashed edges are conditional.
- **Counts:** counts show how often a node ran (×N on the node) and how often a hop was taken (×N on the edge).
- **Live:** the node running right now pulses amber, and the hop just taken animates. Failed nodes are red.
- **Subgraphs:** subgraph nodes (▸) open that subagent's own graph. The breadcrumb leads back up. Selecting an agent on the canvas also switches the panel to its graph.
- **Replay:** scrubbing the timeline replays the graph state too.

Without `graph=`, the panel still draws every node and hop that ran, but it can't show routes that weren't taken.

Superstep attribution: a node's `from` is the set of nodes that ran in the previous superstep, narrowed to those with a declared edge into it when `graph=` is given. With parallel branches of different lengths and no `graph=`, a hop can be attributed to a sibling branch.

## Limitations

- **Coding-agent panels:** panels built for coding agents, such as file attention and permission requests, stay empty.

## Tests

```bash
pip install -e 'adapters/langgraph[dev]'
pytest adapters/langgraph/tests
```
