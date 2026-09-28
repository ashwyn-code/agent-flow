import { test } from 'node:test'
import { strict as assert } from 'node:assert'

import { aggregateGraph, baseName, quantile } from '../web/lib/multi-run'
import type { SimulationEvent } from '../web/lib/agent-types'

type Row = [number, string, Record<string, unknown>]
const events = (rows: Row[]): SimulationEvent[] => rows.map(([time, type, payload]) => ({ time, type, payload } as SimulationEvent))

const STRUCTURE = {
  agent: 'router',
  nodes: [
    { id: '__start__', kind: 'start' }, { id: 'classify', kind: 'node' },
    { id: 'billing', kind: 'node' }, { id: 'tech', kind: 'node' }, { id: '__end__', kind: 'end' },
  ],
  edges: [
    { source: '__start__', target: 'classify', conditional: false },
    { source: 'classify', target: 'billing', conditional: true, label: 'money' },
    { source: 'classify', target: 'tech', conditional: true, label: 'bug' },
    { source: 'billing', target: '__end__', conditional: false },
    { source: 'tech', target: '__end__', conditional: false },
  ],
}

/** One run of the router graph taking `route`, spending `took` seconds there. */
function run(route: 'billing' | 'tech', took: number, opts: { fail?: boolean; name?: string; tokens?: number } = {}): Row[] {
  const g = opts.name ?? 'router'
  return [
    [0, 'agent_spawn', { name: g, isMain: g === 'router' }],
    [0, 'graph_structure', { ...STRUCTURE, agent: g }],
    [0, 'node_start', { agent: g, node: 'classify', from: ['__start__'] }],
    [1, 'node_end', { agent: g, node: 'classify' }],
    [1, 'node_start', { agent: g, node: route, from: ['classify'] }],
    [1, 'agent_spawn', { name: `${route}-agent`, parent: g }],
    [1.5, 'context_update', { agent: `${route}-agent`, tokens: opts.tokens ?? 1000 }],
    [1.5, 'tool_call_start', { agent: `${route}-agent`, tool: 'lookup' }],
    [1 + took, 'tool_call_end', { agent: `${route}-agent`, tool: 'lookup', ...(opts.fail ? { isError: true } : {}) }],
    [1 + took, 'agent_complete', { name: `${route}-agent` }],
    [1 + took, 'node_end', { agent: g, node: route }],
    [1 + took, 'node_start', { agent: g, node: '__end__', from: [route] }],
    [1 + took, 'agent_complete', { name: g }],
  ]
}

test('helpers', () => {
  assert.equal(baseName('research_team #3'), 'research_team')
  assert.equal(baseName('plain'), 'plain')
  assert.equal(quantile([5, 1, 3, 2, 4], 0.5), 3)
  assert.equal(quantile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95), 10)
  assert.equal(quantile([], 0.5), 0)
})

test('routes as a share of runs, node time distributions, error rates', () => {
  const sessions = [
    events(run('billing', 1)),
    events(run('billing', 2, { tokens: 3000 })),
    events(run('billing', 9, { fail: true })),
    events(run('tech', 4)),
    events([[0, 'agent_spawn', { name: 'unrelated', isMain: true }]]),   // didn't run this graph
  ]
  const agg = aggregateGraph(sessions, 'router', { costOf: tokens => tokens / 1000 })!
  assert.equal(agg.runs, 4)
  assert.equal(agg.sessions, 4)
  assert.deepEqual(agg.edges.get('classify->billing'), { runs: 3, share: 0.75 })
  assert.deepEqual(agg.edges.get('classify->tech'), { runs: 1, share: 0.25 })
  assert.equal(agg.edges.get('__start__->classify')!.share, 1)

  const billing = agg.nodes.get('billing')!
  assert.equal(billing.share, 0.75)
  assert.equal(billing.timeP50, 2)
  assert.equal(billing.timeP95, 9)
  assert.equal(billing.tokensP50, 1000)
  assert.ok(Math.abs(billing.costMean - 5 / 3) < 1e-9)
  assert.equal(billing.errorRuns, 1)
  assert.equal(billing.errorShare, 0.25)
  assert.equal(agg.nodes.get('classify')!.share, 1)

  // The union graph keeps the declared structure and counts every traversal
  assert.equal(agg.graph.hasStructure, true)
  assert.equal(agg.graph.edges['classify->billing'].traversals, 3)
  assert.equal(agg.graph.edges['classify->billing'].label, 'money')
  assert.ok(Object.values(agg.graph.nodes).every(n => n.running === 0))
})

test('parallel instances of a subgraph in one session each count as a run', () => {
  const session = events([
    ...run('billing', 1, { name: 'router' }),
    ...run('tech', 2, { name: 'router #2' }),
    ...run('tech', 3, { name: 'router #3' }),
  ])
  const agg = aggregateGraph([session], 'router #2')!
  assert.equal(agg.name, 'router')
  assert.equal(agg.runs, 3)
  assert.equal(agg.sessions, 1)
  assert.equal(agg.edges.get('classify->tech')!.runs, 2)
})

test('no runs of the graph: nothing to aggregate', () => {
  assert.equal(aggregateGraph([events([[0, 'agent_spawn', { name: 'x', isMain: true }]])], 'router'), null)
})
