import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import * as fs from 'fs'
import * as path from 'path'
import * as zlib from 'zlib'

import { analyzeRun, type RunAnalysis } from '../web/lib/run-analysis'
import type { SimulationEvent } from '../web/lib/agent-types'
import { convertSpans } from './otel/convert'
import { spansFromText } from './otel/otlp'

type Row = [number, string, Record<string, unknown>]
const events = (rows: Row[]): SimulationEvent[] => rows.map(([time, type, payload]) => ({ time, type, payload } as SimulationEvent))

/** Real runs: the recorded OpenTelemetry fixtures, converted to events. */
function fixtureEvents(name: string): SimulationEvent[] {
  const text = zlib.gunzipSync(fs.readFileSync(path.join(__dirname, 'otel', 'fixtures', `${name}.otlp.jsonl.gz`))).toString('utf8')
  return convertSpans(spansFromText(text))[0].events as SimulationEvent[]
}

/** The critical path tiles the main agent's run: no gaps, no overlaps. */
function checkPathTiles(a: RunAnalysis): void {
  const main = a.lanes.find(l => l.isMain)!
  const from = main.intervals[0].start
  const to = Math.max(...main.intervals.map(i => i.end ?? a.end))
  assert.ok(a.critical.length > 0)
  assert.ok(Math.abs(a.critical[0].start - from) < 1e-6, 'starts with the run')
  assert.ok(Math.abs(a.critical[a.critical.length - 1].end - to) < 1e-6, 'ends with the run')
  for (let i = 1; i < a.critical.length; i++) {
    assert.ok(Math.abs(a.critical[i].start - a.critical[i - 1].end) < 1e-6, `segment ${i} follows segment ${i - 1}`)
  }
  assert.ok(Math.abs(a.breakdown.tools + a.breakdown.self - (to - from)) < 1e-6)
}

const RUN: Row[] = [
  [0, 'agent_spawn', { name: 'lead', isMain: true }],
  [1, 'tool_call_start', { agent: 'lead', tool: 'plan' }],
  [2, 'tool_call_end', { agent: 'lead', tool: 'plan' }],
  [2, 'agent_spawn', { name: 'fast', parent: 'lead' }],
  [2, 'agent_spawn', { name: 'slow', parent: 'lead' }],
  [2.5, 'tool_call_start', { agent: 'slow', tool: 'search', args: 'a' }],
  [2.5, 'tool_call_start', { agent: 'slow', tool: 'search', args: 'b' }],
  [3, 'agent_complete', { name: 'fast' }],
  [6, 'tool_call_end', { agent: 'slow', tool: 'search' }],
  [7, 'tool_call_end', { agent: 'slow', tool: 'search', isError: true, errorMessage: 'timeout' }],
  [8, 'agent_complete', { name: 'slow' }],
  [10, 'agent_complete', { name: 'lead' }],
]

test('lanes in tree order, parallel tool calls on their own rows', () => {
  const a = analyzeRun(events(RUN), 10)
  assert.deepEqual(a.lanes.map(l => [l.name, l.depth]), [['lead', 0], ['fast', 1], ['slow', 1]])
  const slow = a.lanes.find(l => l.name === 'slow')!
  assert.equal(slow.rows, 2)
  assert.deepEqual(slow.tools.map(t => [t.row, t.end, t.error]), [[0, 6, undefined], [1, 7, 'timeout']])
  assert.equal(slow.errors, 1)
})

test('the critical path goes through the slow branch and its last tool call', () => {
  const a = analyzeRun(events(RUN), 10)
  checkPathTiles(a)
  assert.deepEqual(a.critical.map(s => [s.lane, s.kind, s.start, s.end]), [
    ['lead', 'self', 0, 1],
    ['lead', 'tool', 1, 2],
    ['slow', 'self', 2, 2.5],
    ['slow', 'tool', 2.5, 7],
    ['slow', 'self', 7, 8],
    ['lead', 'self', 8, 10],
  ])
  assert.ok(!a.criticalLanes.has('fast'))
  assert.equal(a.criticalToolIds.size, 2)
  assert.deepEqual(a.breakdown, { tools: 5.5, self: 4.5 })
  assert.deepEqual(a.topCritical[0], { lane: 'slow', label: 'search', kind: 'tool', duration: 4.5 })
})

test('an agent called as a tool: the path goes into the agent', () => {
  const a = analyzeRun(events([
    [0, 'agent_spawn', { name: 'boss', isMain: true }],
    [1, 'tool_call_start', { agent: 'boss', tool: 'ask_expert' }],
    [1.1, 'agent_spawn', { name: 'expert', parent: 'boss' }],
    [2, 'tool_call_start', { agent: 'expert', tool: 'lookup' }],
    [4, 'tool_call_end', { agent: 'expert', tool: 'lookup' }],
    [4.5, 'agent_complete', { name: 'expert' }],
    [4.6, 'tool_call_end', { agent: 'boss', tool: 'ask_expert' }],
    [5, 'agent_complete', { name: 'boss' }],
  ]), 5)
  checkPathTiles(a)
  assert.deepEqual(a.critical.map(s => [s.lane, s.kind, s.label]), [
    ['boss', 'self', 'boss'],
    ['boss', 'tool', 'ask_expert'],      // the call before the agent starts
    ['expert', 'self', 'expert'],
    ['expert', 'tool', 'lookup'],
    ['expert', 'self', 'expert'],
    ['boss', 'tool', 'ask_expert'],      // and after it returns
    ['boss', 'self', 'boss'],
  ])
  assert.ok(a.criticalLanes.has('expert'))
})

test('follows the clock: a run in progress, or scrubbed back', () => {
  const partial = analyzeRun(events(RUN), 4)
  const slow = partial.lanes.find(l => l.name === 'slow')!
  assert.equal(slow.intervals[0].end, undefined, 'still running at t=4')
  assert.equal(slow.tools[0].end, undefined)
  assert.equal(partial.end, 4)
  checkPathTiles(partial)
})

test('graph node metrics: time, tokens, cost and errors per node', () => {
  const rows: Row[] = [
    [0, 'agent_spawn', { name: 'flow', isMain: true }],
    [0, 'node_start', { agent: 'flow', node: 'research', from: ['__start__'] }],
    [0, 'agent_spawn', { name: 'research', parent: 'flow' }],
    [0.5, 'agent_spawn', { name: 'helper', parent: 'research' }],
    [1, 'context_update', { agent: 'helper', tokens: 500 }],
    [1, 'model_detected', { agent: 'helper', model: 'm-big' }],
    [1.5, 'tool_call_start', { agent: 'helper', tool: 'x' }],
    [2, 'tool_call_end', { agent: 'helper', tool: 'x', isError: true }],
    [2, 'agent_complete', { name: 'helper' }],
    [2.5, 'context_update', { agent: 'research', tokens: 1000 }],
    [3, 'agent_complete', { name: 'research' }],
    [3, 'node_end', { agent: 'flow', node: 'research' }],
    [3, 'node_start', { agent: 'flow', node: 'route', from: ['research'] }],  // a plain function node
    [3.5, 'node_end', { agent: 'flow', node: 'route', error: 'no route' }],
    [3.5, 'node_start', { agent: 'flow', node: 'research', from: ['route'] }],
    [3.5, 'agent_spawn', { name: 'research', parent: 'flow' }],
    [4, 'context_update', { agent: 'research', tokens: 200 }],
    [5, 'agent_complete', { name: 'research' }],
    [5, 'node_end', { agent: 'flow', node: 'research' }],
    [5, 'node_start', { agent: 'flow', node: '__end__', from: ['research'] }],
    [5, 'agent_complete', { name: 'flow' }],
  ]
  const a = analyzeRun(events(rows), 5, { costOf: (tokens, model) => tokens * (model === 'm-big' ? 0.01 : 0.001) })
  const m = a.nodeMetrics.get('flow')!
  assert.deepEqual([...m.keys()], ['research', 'route'])
  const research = m.get('research')!
  assert.equal(research.runs, 2)
  assert.equal(research.time, 4.5)
  assert.equal(research.tokens, 1700)   // both runs, including the helper they spawned
  assert.ok(Math.abs(research.cost - (1.0 + 5 + 0.2)) < 1e-9)
  assert.equal(research.errors, 1)
  assert.deepEqual(m.get('route'), { runs: 1, time: 0.5, tokens: 0, cost: 0, errors: 1 })
})

test('real runs: Google ADK and Agent Framework traces', () => {
  const adk = analyzeRun(fixtureEvents('google-adk'), Infinity)
  checkPathTiles(adk)
  assert.equal(adk.lanes[0].name, 'trip_planner')
  const top = adk.nodeMetrics.get('trip_planner')!
  assert.deepEqual([...top.keys()], ['intake', 'research', 'itinerary_loop', 'booking'])
  assert.equal(top.get('research')!.errors, 1)   // search_flights failed in one branch
  assert.ok(top.get('itinerary_loop')!.tokens > top.get('intake')!.tokens)
  assert.equal(adk.nodeMetrics.get('itinerary_loop')!.get('planner')!.runs, 2)

  const maf = analyzeRun(fixtureEvents('agent-framework'), Infinity)
  checkPathTiles(maf)
  const nodes = maf.nodeMetrics.get('incident_response')!
  assert.equal(nodes.get('commander')!.runs, 2)
  assert.equal(nodes.get('metrics_analyst')!.errors, 1)
  assert.equal(nodes.get('intake')!.tokens, 0)
  // Lanes list the parallel analysts under the workflow
  const names = maf.lanes.map(l => l.name)
  assert.ok(names.indexOf('log_analyst') > names.indexOf('incident_response'))
})
