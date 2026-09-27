import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import * as fs from 'fs'
import * as path from 'path'
import * as zlib from 'zlib'

import type { AgentEvent } from '../../extension/src/protocol'
import { convertSpans, convertTrace, type ConvertedTrace } from './convert'
import { spansFromText, type Span } from './otlp'

// Fixtures: the adapters' offline demos run with OpenTelemetry tracing on
// (each framework's own spans, or OpenInference's instrumentation), exported
// as OTLP JSON, scrubbed and gzipped. See fixtures/README.md.
const FIXTURES = path.join(__dirname, 'fixtures')

function load(name: string): ConvertedTrace {
  const text = zlib.gunzipSync(fs.readFileSync(path.join(FIXTURES, `${name}.otlp.jsonl.gz`))).toString('utf8')
  const traces = convertSpans(spansFromText(text))
  assert.equal(traces.length, 1, `${name}: one trace`)
  return traces[0]
}

const KNOWN_TYPES = new Set([
  'agent_spawn', 'agent_complete', 'message', 'context_update', 'model_detected',
  'tool_call_start', 'tool_call_end', 'subagent_dispatch', 'subagent_return',
  'graph_structure', 'node_start', 'node_end',
])

type P = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any

const of = (events: AgentEvent[], type: string): P[] => events.filter(e => e.type === type).map(e => e.payload as P)

function parents(events: AgentEvent[]): Map<string, string | null> {
  const out = new Map<string, string | null>()
  for (const p of of(events, 'agent_spawn')) if (!out.has(p.name)) out.set(p.name, p.parent ?? null)
  return out
}

function edges(events: AgentEvent[], agent: string): Set<string> {
  const g = of(events, 'graph_structure').find(p => p.agent === agent)
  assert.ok(g, `graph for ${agent}`)
  return new Set(g!.edges.map((e: P) => `${e.source}->${e.target}${e.conditional ? '?' : ''}${e.label ? `[${e.label}]` : ''}`))
}

function hops(events: AgentEvent[], agent: string): Set<string> {
  const out = new Set<string>()
  for (const p of of(events, 'node_start')) if (p.agent === agent) for (const f of p.from) out.add(`${f}->${p.node}`)
  return out
}

/** Invariants the webview relies on, for any converted trace. */
function checkWellFormed(trace: ConvertedTrace): void {
  const { events } = trace
  assert.ok(events.every(e => KNOWN_TYPES.has(e.type)), 'known event types')
  for (let i = 1; i < events.length; i++) assert.ok(events[i].time >= events[i - 1].time, 'sorted by time')
  assert.equal(of(events, 'agent_spawn').filter(p => p.isMain).length, 1, 'one main agent')
  assert.equal(events[0].type, 'agent_spawn')
  assert.deepEqual(events[events.length - 1], { ...events[events.length - 1], type: 'agent_complete', payload: { name: trace.name } })

  const running = new Set<string>()
  const openTools = new Map<string, number>()
  const openNodes = new Map<string, number>()
  for (const e of events) {
    const p = e.payload as P
    const agent = p.agent ?? p.parent
    if (e.type === 'agent_spawn') {
      assert.ok(!running.has(p.name), `${p.name} spawned twice at once`)
      if (p.parent) assert.ok(running.has(p.parent), `parent ${p.parent} of ${p.name} is running`)
      running.add(p.name)
    } else if (e.type === 'agent_complete') {
      assert.ok(running.delete(p.name), `${p.name} completes after spawning`)
    } else if (agent !== undefined) {
      assert.ok(running.has(agent), `${e.type} for running agent ${agent}`)
    }
    const toolKey = `${p.agent}/${p.tool}`
    if (e.type === 'tool_call_start') openTools.set(toolKey, (openTools.get(toolKey) ?? 0) + 1)
    if (e.type === 'tool_call_end') {
      assert.ok((openTools.get(toolKey) ?? 0) > 0, `tool ${toolKey} ends after it starts`)
      openTools.set(toolKey, openTools.get(toolKey)! - 1)
    }
    const nodeKey = `${p.agent}/${p.node}`
    if (e.type === 'node_start' && p.node !== '__end__') openNodes.set(nodeKey, (openNodes.get(nodeKey) ?? 0) + 1)
    if (e.type === 'node_end') {
      assert.ok((openNodes.get(nodeKey) ?? 0) > 0, `node ${nodeKey} ends after it starts`)
      openNodes.set(nodeKey, openNodes.get(nodeKey)! - 1)
    }
  }
  assert.equal(running.size, 0, 'every agent completes')
  assert.ok([...openTools.values()].every(n => n === 0), 'every tool call ends')
  assert.ok([...openNodes.values()].every(n => n === 0), 'every node ends')
}

test('Strands (GenAI conventions): graph, swarm and agents-as-tools', () => {
  const trace = load('strands')
  checkWellFormed(trace)
  const { events } = trace
  assert.equal(trace.name, 'graph')
  assert.match(of(events, 'message')[0].content, /^Plan the launch of our AI analytics product/)
  const tree = parents(events)
  assert.equal(tree.get('planner'), 'graph')
  assert.equal(tree.get('architect'), 'swarm')
  assert.equal(tree.get('statistician'), 'survey_analyst')   // an agent called as a tool
  assert.equal(tree.get('analyst_acme'), 'competitor_scan')
  // The top graph fans out after the planner and loops reviewer -> strategist
  const top = edges(events, 'graph')
  for (const e of ['__start__->planner', 'planner->swarm', 'planner->graph', 'planner->competitor_scan', 'reviewer->strategist', 'publisher->__end__']) {
    assert.ok(top.has(e), e)
  }
  // A swarm is drawn as the handoffs that happened
  assert.deepEqual(hops(events, 'swarm'), new Set(['__start__->architect', 'architect->security', 'security->architect', 'architect->performance', 'performance->__end__']))
  // Nested graphs are subgraph nodes linked to their agent
  const g = of(events, 'graph_structure').find(p => p.agent === 'graph')!
  assert.deepEqual(g.nodes.filter((n: P) => n.kind === 'subgraph').map((n: P) => [n.id, n.child]), [['swarm', 'swarm'], ['graph', 'graph #2']])
  const failed = of(events, 'tool_call_end').filter(p => p.isError)
  assert.deepEqual(failed.map(p => [p.agent, p.tool, p.errorMessage]), [['analyst_globex', 'pricing_page', 'pricing page returned 503']])
  const thinking = of(events, 'message').find(p => p.role === 'thinking' && p.agent === 'planner')
  assert.equal(thinking?.content, 'Three independent threads; run them concurrently.')
  assert.ok(of(events, 'model_detected').some(p => p.agent === 'planner' && p.model === 'demo-claude-opus-4-1'))
})

test('Agent Framework: declared workflow graph with conditions, nested workflow', () => {
  const trace = load('agent-framework')
  checkWellFormed(trace)
  const { events } = trace
  assert.equal(trace.name, 'incident_response')
  const declared = edges(events, 'incident_response')
  for (const e of ['__start__->intake', 'triage->dispatch?[is_high]', 'triage->auto_ack?[default]', 'dispatch->log_analyst', 'log_analyst->correlate', 'remediation->commander?', 'remediation->postmortem?']) {
    assert.ok(declared.has(e), e)
  }
  const nodes = of(events, 'graph_structure').find(p => p.agent === 'incident_response')!.nodes
  assert.ok(nodes.some((n: P) => n.id === 'auto_ack'), 'a declared node that never ran is still drawn')
  assert.deepEqual(nodes.find((n: P) => n.id === 'remediation'), { id: 'remediation', label: 'remediation', kind: 'subgraph', child: 'remediation' })
  // Function executors are nodes without agents; agent executors are agents
  const tree = parents(events)
  assert.ok(!tree.has('intake') && !tree.has('dispatch'))
  assert.equal(tree.get('triage'), 'incident_response')
  assert.equal(tree.get('planner'), 'remediation')
  assert.equal(tree.get('git_historian'), 'deploy_auditor')
  // The commander <-> remediation loop ran twice
  const taken = hops(events, 'incident_response')
  assert.ok(taken.has('remediation->commander') && taken.has('commander->remediation'))
  assert.equal(of(events, 'node_start').filter(p => p.agent === 'incident_response' && p.node === 'commander').length, 2)
  assert.ok(taken.has('dispatch->log_analyst') && taken.has('dispatch->metrics_analyst') && taken.has('dispatch->deploy_auditor'))
  const failed = of(events, 'tool_call_end').find(p => p.isError)!
  assert.equal(failed.tool, 'query_metrics')
  assert.match(failed.errorMessage, /metrics store timed out/)
  const search = of(events, 'tool_call_start').find(p => p.tool === 'search_logs')!
  assert.deepEqual(search.inputData, { query: 'checkout 5xx' })
})

test('Google ADK: sequential, parallel and loop agents become graphs', () => {
  const trace = load('google-adk')
  checkWellFormed(trace)
  const { events } = trace
  assert.equal(trace.name, 'trip_planner')
  assert.match(of(events, 'message')[0].content, /^Plan a 4-day trip to Lisbon/)
  assert.deepEqual(hops(events, 'trip_planner'), new Set(['__start__->intake', 'intake->research', 'research->itinerary_loop', 'itinerary_loop->booking', 'booking->__end__']))
  assert.deepEqual(hops(events, 'research'), new Set(['__start__->flights', '__start__->hotels', '__start__->weather', 'flights->__end__', 'hotels->__end__', 'weather->__end__']))
  assert.deepEqual(hops(events, 'itinerary_loop'), new Set(['__start__->planner', 'planner->critic', 'critic->planner', 'critic->__end__']))
  assert.equal(parents(events).get('budget_analyst'), 'planner')
  // ADK tools report failures as {"status": "error"} results
  const failed = of(events, 'tool_call_end').filter(p => p.isError)
  assert.deepEqual(failed.map(p => [p.tool, p.errorMessage]), [['search_flights', 'fares API timed out']])
  assert.ok(!of(events, 'tool_call_start').some(p => p.tool === '(merged tools)'))
})

test('OpenAI Agents (OpenInference): handoffs, agents as tools, guardrails', () => {
  const trace = load('openai-agents-openinference')
  checkWellFormed(trace)
  const { events } = trace
  assert.equal(trace.name, 'Support desk')
  assert.deepEqual(hops(events, 'Agent workflow'), new Set(['__start__->Triage', 'Triage->Tech Support', 'Tech Support->Billing', 'Billing->__end__']))
  const tree = parents(events)
  assert.equal(tree.get('Sentiment Analyst'), 'Support desk')
  assert.equal(tree.get('Log Analyst'), 'Tech Support')
  assert.equal(tree.get('DB Inspector'), 'Log Analyst')
  const tools = of(events, 'tool_call_start').map(p => `${p.agent}:${p.tool}`)
  assert.ok(tools.includes('Triage:guardrail: pii_filter'))
  assert.ok(tools.includes('Triage:handoff to Tech Support'))
  assert.ok(of(events, 'tool_call_end').some(p => p.tool === 'check_service_status' && p.isError))
  const thinking = of(events, 'message').find(p => p.agent === 'Triage' && p.role === 'thinking')
  assert.equal(thinking?.content, 'Outage first, then the credit: Tech Support.')
})

test('LangGraph (OpenInference): nodes, subgraphs, fan-out/fan-in and loops', () => {
  const trace = load('langgraph-openinference')
  checkWellFormed(trace)
  const { events } = trace
  assert.equal(trace.name, 'orchestrator')
  const top = hops(events, 'orchestrator')
  for (const e of ['__start__->planner', 'planner->research_team', 'research_team->synthesize', 'writer->critic', 'critic->writer', 'publish->__end__']) {
    assert.ok(top.has(e), e)
  }
  // Three research teams ran at once (Send fan-out), each its own subgraph
  const teams = [...parents(events)].filter(([, parent]) => parent === 'orchestrator').map(([name]) => name)
  assert.deepEqual(teams.filter(n => n.startsWith('research_team')).sort(), ['research_team', 'research_team #2', 'research_team #3'])
  assert.deepEqual(hops(events, 'research_team'), new Set([
    '__start__->lead', 'lead->data_analyst', 'lead->paper_researcher', 'lead->web_researcher',
    'data_analyst->merge', 'paper_researcher->merge', 'web_researcher->merge', 'merge->__end__',
  ]))
  assert.deepEqual(hops(events, 'data_analyst'), new Set(['__start__->agent', 'agent->tools', 'tools->agent', 'agent->__end__']))
  const failed = of(events, 'tool_call_end').find(p => p.isError)!
  assert.equal(failed.tool, 'run_sql')
  assert.equal(failed.errorMessage, "TimeoutError('warehouse query timed out after 30s')")
  assert.match(of(events, 'message')[0].content, /^Write a board-ready report/)
})

// ─── Synthetic traces ────────────────────────────────────────────────────────

let nextId = 1
function span(name: string, start: number, end: number, parent: Span | null, attributes: Span['attributes'] = {}, extra: Partial<Span> = {}): Span {
  const id = (nextId++).toString(16).padStart(16, '0')
  const t0 = BigInt('1700000000000000000')
  return {
    traceId: 'f'.repeat(32), spanId: id, parentSpanId: parent?.spanId ?? '', name,
    start: t0 + BigInt(Math.round(start * 1e9)), end: t0 + BigInt(Math.round(end * 1e9)),
    attributes, events: [], statusCode: 0, statusMessage: '', resource: { 'service.name': 'checkout' }, scope: '',
    ...extra,
  }
}
const agent = (name: string, start: number, end: number, parent: Span | null) =>
  span(`invoke_agent ${name}`, start, end, parent, { 'gen_ai.operation.name': 'invoke_agent', 'gen_ai.agent.name': name })
const chat = (start: number, end: number, parent: Span, text: string) =>
  span('chat', start, end, parent, {
    'gen_ai.operation.name': 'chat', 'gen_ai.request.model': 'm1',
    'gen_ai.usage.input_tokens': 100, 'gen_ai.usage.output_tokens': 20,
    'gen_ai.output.messages': JSON.stringify([{ role: 'assistant', parts: [{ type: 'text', content: text }] }]),
  })

test('an LLM agent calling sub-agents concurrently: sub-agents, unique names', () => {
  const root = agent('lead', 0, 10, null)
  const spans = [
    root, chat(0, 1, root, 'split it'),
    agent('worker', 1, 5, root), agent('worker', 2, 6, root), agent('worker', 7, 8, root),
  ]
  const trace = convertTrace(spans)!
  checkWellFormed(trace)
  // lead calls a model, so it's not a graph: workers are plain sub-agents
  assert.equal(of(trace.events, 'graph_structure').length, 0)
  assert.deepEqual(of(trace.events, 'agent_spawn').map(p => p.name), ['lead', 'worker', 'worker #2', 'worker'])
  assert.deepEqual(of(trace.events, 'context_update')[0], { agent: 'lead', tokens: 120 })
})

test('clock skew between services cannot put a child outside its parent', () => {
  const root = agent('frontend', 5, 6, null)
  const child = agent('backend', 4, 9, root)   // another host's clock is off
  const trace = convertTrace([root, child])!
  checkWellFormed(trace)
  const spawn = trace.events.find(e => e.type === 'agent_spawn' && (e.payload as P).name === 'backend')!
  const complete = trace.events.find(e => e.type === 'agent_complete' && (e.payload as P).name === 'backend')!
  assert.equal(spawn.time, 0)
  assert.equal(complete.time, 1)
})

test('several roots or a bare model call still get a main agent', () => {
  const a = agent('a', 0, 1, null)
  const b = agent('b', 0.5, 2, null)
  const trace = convertTrace([a, b])!
  checkWellFormed(trace)
  assert.equal(trace.name, 'checkout')   // service.name
  assert.deepEqual([...parents(trace.events)], [['checkout', null], ['a', 'checkout'], ['b', 'checkout']])

  const request = span('POST /chat', 0, 1, null)
  const single = convertTrace([request, chat(0, 1, request, 'hello')])!
  checkWellFormed(single)
  assert.equal(single.name, 'POST /chat')
  assert.ok(of(single.events, 'message').some(p => p.role === 'assistant' && p.content === 'hello'))
})

test('traces with nothing agent-like are skipped', () => {
  const build = span('workflow.build', 0, 1, null, { 'workflow.id': 'w1' })
  assert.equal(convertTrace([build, span('db.query', 0, 0.5, build)]), null)
  assert.equal(convertTrace([]), null)
})

test('long content is clipped and tool errors come from exception events', () => {
  const root = agent('a', 0, 3, null)
  const tool = span('execute_tool fetch', 1, 2, root,
    { 'gen_ai.operation.name': 'execute_tool', 'gen_ai.tool.name': 'fetch', 'gen_ai.tool.call.arguments': JSON.stringify({ q: 'x'.repeat(5000) }) },
    { statusCode: 2, events: [{ time: BigInt(0), name: 'exception', attributes: { 'exception.message': 'connection reset' } }] })
  const trace = convertTrace([root, tool])!
  const [start] = of(trace.events, 'tool_call_start')
  assert.ok(start.args.length <= 1000)
  const [end] = of(trace.events, 'tool_call_end')
  assert.equal(end.isError, true)
  assert.equal(end.errorMessage, 'connection reset')
})
