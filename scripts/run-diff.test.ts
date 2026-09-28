import { test } from 'node:test'
import { strict as assert } from 'node:assert'

import { diffRuns, sessionsWithGraph } from '../web/lib/run-diff'
import type { SimulationEvent } from '../web/lib/agent-types'

type Row = [number, string, Record<string, unknown>]
const events = (rows: Row[]): SimulationEvent[] => rows.map(([time, type, payload]) => ({ time, type, payload } as SimulationEvent))

/** A run of a small routing graph: classify, then `route`, then reply. */
function run(route: string, took: number, opts: { fail?: boolean; tokens?: number; name?: string } = {}): Row[] {
  const g = opts.name ?? 'router'
  const end = 1 + took + 0.5
  return [
    [0, 'agent_spawn', { name: g, isMain: g === 'router' }],
    [0, 'graph_structure', { agent: g, nodes: [{ id: '__start__', kind: 'start' }, { id: '__end__', kind: 'end' }], edges: [] }],
    [0, 'node_start', { agent: g, node: 'classify', from: ['__start__'] }],
    [1, 'node_end', { agent: g, node: 'classify' }],
    [1, 'node_start', { agent: g, node: route, from: ['classify'] }],
    [1, 'agent_spawn', { name: `${g}/${route}`, parent: g }],
    [1.2, 'context_update', { agent: `${g}/${route}`, tokens: opts.tokens ?? 1000 }],
    [1.2, 'tool_call_start', { agent: `${g}/${route}`, tool: 'lookup' }],
    [1 + took, 'tool_call_end', { agent: `${g}/${route}`, tool: 'lookup', ...(opts.fail ? { isError: true } : {}) }],
    [1 + took, 'agent_complete', { name: `${g}/${route}` }],
    [1 + took, 'node_end', { agent: g, node: route }],
    [1 + took, 'node_start', { agent: g, node: 'reply', from: [route] }],
    [end, 'node_end', { agent: g, node: 'reply' }],
    [end, 'node_start', { agent: g, node: '__end__', from: ['reply'] }],
    [end, 'agent_complete', { name: g }],
  ]
}

test('a changed route: new and gone edges and nodes, with deltas', () => {
  const baseline = events(run('billing', 1))
  const current = events(run('tech', 4, { fail: true, tokens: 5000 }))
  const d = diffRuns(current, baseline, 'router', { costOf: tokens => tokens / 1000 })!
  assert.equal(d.name, 'router')
  assert.equal(d.edges.get('classify->tech'), 'added')
  assert.equal(d.edges.get('tech->reply'), 'added')
  assert.equal(d.edges.get('classify->billing'), 'removed')
  assert.equal(d.edges.get('__start__->classify'), 'both')
  assert.equal(d.nodes.get('tech')!.status, 'added')
  assert.equal(d.nodes.get('billing')!.status, 'removed')
  assert.equal(d.nodes.get('billing')!.time, -1)
  assert.equal(d.nodes.get('tech')!.time, 4)
  assert.equal(d.nodes.get('tech')!.errors, 1)
  assert.equal(d.nodes.get('classify')!.status, 'both')
  assert.equal(d.nodes.get('classify')!.time, 0)
  // Both routes appear in the union graph
  assert.ok(d.graph.nodes.tech && d.graph.nodes.billing)
  // Totals
  assert.equal(d.baseline.duration, 2.5)
  assert.equal(d.current.duration, 5.5)
  assert.equal(d.current.errors, 1)
  assert.equal(d.baseline.errors, 0)
  assert.equal(d.current.tokens - d.baseline.tokens, 4000)
  assert.ok(Math.abs(d.current.cost - d.baseline.cost - 4) < 1e-9)
  assert.equal(d.current.toolCalls, 1)
})

test('same route, slower: only deltas', () => {
  const d = diffRuns(events(run('billing', 3)), events(run('billing', 1)), 'router')!
  assert.ok([...d.edges.values()].every(s => s === 'both'))
  assert.deepEqual([...d.nodes.values()].map(n => n.status), ['both', 'both', 'both'])
  assert.equal(d.nodes.get('billing')!.time, 2)
})

test('a parallel instance compares against the same-named instance, ignoring its siblings', () => {
  const session = events([...run('billing', 1, { name: 'router #2' }), ...run('tech', 9, { name: 'router #3' })])
  const d = diffRuns(session, events(run('billing', 2)), 'router #2')!
  assert.equal(d.edges.get('classify->billing'), 'both')
  assert.equal(d.edges.has('classify->tech'), false)
  assert.equal(d.nodes.get('billing')!.time, -1)
})

test('finding sessions to compare with', () => {
  const sessions = new Map<string, SimulationEvent[]>([
    ['a', events(run('billing', 1))],
    ['b', events([[0, 'agent_spawn', { name: 'other', isMain: true }]])],
    ['c', events(run('tech', 1))],
  ])
  assert.deepEqual(sessionsWithGraph(sessions, 'router', 'a'), ['c'])
  assert.equal(diffRuns(sessions.get('a')!, sessions.get('b')!, 'router'), null)
})
