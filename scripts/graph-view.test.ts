import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import { layoutGraph, findBackEdges } from '../web/lib/graph-layout'
import { processEvent, type ProcessEventContext } from '../web/hooks/simulation/process-event'
import { createEmptyState, type SimulationState } from '../web/hooks/simulation/types'
import type { SimulationEvent, AgentGraph } from '../web/lib/agent-types'

const ctx: ProcessEventContext = {
  syncForceSimulation() {},
  findToolSlot: () => ({ x: 0, y: 0 }),
  getContextWindowSize: () => 200_000,
  blockIdCounter: { current: 0 },
  skipForceSync: true,
}

function run(events: Array<[string, Record<string, unknown>]>): SimulationState {
  let state = createEmptyState()
  events.forEach(([type, payload], i) => {
    const event = { time: i, type, payload } as SimulationEvent
    state = processEvent(event, { ...state, currentTime: i }, ctx)
  })
  return state
}

const REACT_STRUCTURE = {
  agent: 'researcher',
  nodes: [
    { id: '__start__', label: '__start__', kind: 'start' },
    { id: 'agent', label: 'agent', kind: 'node' },
    { id: 'tools', label: 'tools', kind: 'node' },
    { id: '__end__', label: '__end__', kind: 'end' },
  ],
  edges: [
    { source: '__start__', target: 'agent', conditional: false },
    { source: 'agent', target: 'tools', conditional: true },
    { source: 'agent', target: '__end__', conditional: true },
    { source: 'tools', target: 'agent', conditional: false },
  ],
}

const REACT_RUN: Array<[string, Record<string, unknown>]> = [
  ['agent_spawn', { name: 'researcher', isMain: true }],
  ['graph_structure', REACT_STRUCTURE],
  ['node_start', { agent: 'researcher', node: 'agent', step: 1, from: ['__start__'] }],
  ['node_end', { agent: 'researcher', node: 'agent', step: 1 }],
  ['node_start', { agent: 'researcher', node: 'tools', step: 2, from: ['agent'] }],
  ['node_end', { agent: 'researcher', node: 'tools', step: 2 }],
  ['node_start', { agent: 'researcher', node: 'agent', step: 3, from: ['tools'] }],
]

// ─── Reducer ─────────────────────────────────────────────────────────────────

test('graph_structure declares nodes and edges with conditional flags', () => {
  const graph = run(REACT_RUN.slice(0, 2)).graphs.get('researcher')!
  assert.equal(graph.hasStructure, true)
  assert.deepEqual(graph.order, ['__start__', 'agent', 'tools', '__end__'])
  assert.equal(graph.nodes.__end__.kind, 'end')
  assert.equal(graph.edges['agent->tools'].conditional, true)
  assert.equal(graph.edges['tools->agent'].conditional, false)
  assert.ok(Object.values(graph.edges).every(e => e.declared && e.traversals === 0))
})

test('node_start/node_end count visits, hops and in-flight nodes through a loop', () => {
  const graph = run(REACT_RUN).graphs.get('researcher')!
  assert.equal(graph.nodes.agent.visits, 2)
  assert.equal(graph.nodes.agent.running, 1) // step 3 still in flight
  assert.equal(graph.nodes.tools.visits, 1)
  assert.equal(graph.nodes.tools.running, 0)
  assert.equal(graph.nodes.__start__.visits, 1)
  assert.equal(graph.edges['__start__->agent'].traversals, 1)
  assert.equal(graph.edges['agent->tools'].traversals, 1)
  assert.equal(graph.edges['tools->agent'].traversals, 1)
  assert.equal(graph.edges['agent->__end__'].traversals, 0)
  assert.equal(graph.totalHops, 3)
  assert.equal(graph.lastStep, 3)
})

test('agent_complete clears in-flight nodes', () => {
  const graph = run([...REACT_RUN, ['agent_complete', { name: 'researcher' }]]).graphs.get('researcher')!
  assert.equal(graph.nodes.agent.running, 0)
})

test('without a structure, nodes and edges are built from observed hops (including merges)', () => {
  const graph = run([
    ['agent_spawn', { name: 'g', isMain: true }],
    ['node_start', { agent: 'g', node: 'a', step: 1, from: ['__start__'] }],
    ['node_start', { agent: 'g', node: 'b', step: 2, from: ['a'] }],
    ['node_start', { agent: 'g', node: 'c', step: 2, from: ['a'] }],
    ['node_start', { agent: 'g', node: 'd', step: 3, from: ['b', 'c'] }],
    ['node_start', { agent: 'g', node: '__end__', step: 4, from: ['d'] }],
  ]).graphs.get('g')!
  assert.equal(graph.hasStructure, false)
  assert.deepEqual(graph.order, ['__start__', 'a', 'b', 'c', 'd', '__end__'])
  assert.equal(graph.nodes.__start__.kind, 'start')
  assert.equal(graph.nodes.__end__.kind, 'end')
  assert.equal(graph.nodes.__end__.running, 0)
  assert.deepEqual(Object.keys(graph.edges).sort(), ['__start__->a', 'a->b', 'a->c', 'b->d', 'c->d', 'd->__end__'])
  assert.ok(Object.values(graph.edges).every(e => !e.declared && e.traversals === 1))
})

test('node errors are kept until the node runs again', () => {
  const graph = run([
    ['node_start', { agent: 'g', node: 'a', step: 1, from: ['__start__'] }],
    ['node_end', { agent: 'g', node: 'a', step: 1, error: 'boom' }],
  ]).graphs.get('g')!
  assert.equal(graph.nodes.a.error, 'boom')
})

test('non-graph events leave the graphs map reference unchanged', () => {
  const before = run(REACT_RUN)
  const after = processEvent({ time: 99, type: 'message', payload: { agent: 'researcher', content: 'hi' } }, before, ctx)
  assert.equal(after.graphs, before.graphs)
})

// ─── Layout ──────────────────────────────────────────────────────────────────

function graphOf(structure: typeof REACT_STRUCTURE): AgentGraph {
  return run([['graph_structure', structure]]).graphs.get(structure.agent)!
}

test('findBackEdges marks the edge that closes each cycle', () => {
  const edges = [
    { id: 's->a', source: '__start__', target: 'a' },
    { id: 'a->b', source: 'a', target: 'b' },
    { id: 'b->a', source: 'b', target: 'a' },
    { id: 'b->b', source: 'b', target: 'b' },
    { id: 'b->e', source: 'b', target: '__end__' },
  ]
  assert.deepEqual([...findBackEdges(['__start__', 'a', 'b', '__end__'], edges)].sort(), ['b->a', 'b->b'])
})

test('layout ranks a ReAct loop top-down with the loop routed as a back edge', () => {
  const layout = layoutGraph(graphOf(REACT_STRUCTURE))
  const rank = (id: string) => layout.nodes.get(id)!.rank
  assert.equal(rank('__start__'), 0)
  assert.equal(rank('agent'), 1)
  assert.equal(rank('tools'), 2)
  assert.equal(rank('__end__'), 3) // pinned below everything
  assert.equal(layout.edges.get('tools->agent')!.kind, 'back')
  assert.equal(layout.edges.get('agent->tools')!.kind, 'forward')
  // Loop arc sits to the right of the nodes it spans
  const loop = layout.edges.get('tools->agent')!
  const right = Math.max(layout.nodes.get('agent')!.x + layout.nodes.get('agent')!.w, layout.nodes.get('tools')!.x + layout.nodes.get('tools')!.w)
  assert.ok(loop.labelX > right)
  assert.ok(layout.width > loop.labelX)
})

test('layout places parallel branches side by side and the merge below both', () => {
  const layout = layoutGraph(graphOf({
    agent: 'd',
    nodes: ['__start__', 'a', 'b', 'c', 'd', '__end__'].map(id => ({ id, label: id, kind: id === '__start__' ? 'start' : id === '__end__' ? 'end' : 'node' })),
    edges: [['__start__', 'a'], ['a', 'b'], ['a', 'c'], ['b', 'd'], ['c', 'd'], ['d', '__end__']].map(([source, target]) => ({ source, target, conditional: false })),
  }))
  const b = layout.nodes.get('b')!, c = layout.nodes.get('c')!, d = layout.nodes.get('d')!
  assert.equal(b.rank, c.rank)
  assert.ok(b.x + b.w <= c.x || c.x + c.w <= b.x, 'branches must not overlap')
  assert.equal(d.rank, b.rank + 1)
  assert.equal(layout.edges.get('b->d')!.kind, 'forward')
  assert.equal(layout.edges.get('c->d')!.kind, 'forward')
  // No two nodes overlap anywhere
  const all = [...layout.nodes.values()]
  for (const p of all) for (const q of all) {
    if (p === q) continue
    const overlap = p.x < q.x + q.w && q.x < p.x + p.w && p.y < q.y + q.h && q.y < p.y + p.h
    assert.ok(!overlap, `${p.id} overlaps ${q.id}`)
  }
})

test('layout handles self-loops and graphs without start/end nodes', () => {
  const layout = layoutGraph(graphOf({
    agent: 'x',
    nodes: [{ id: 'poll', label: 'poll', kind: 'node' }],
    edges: [{ source: 'poll', target: 'poll', conditional: true }],
  }))
  assert.equal(layout.edges.get('poll->poll')!.kind, 'self')
  assert.equal(layout.nodes.get('poll')!.rank, 0)
})

test('graphs remember their parent agent, so finished subagents stay reachable', () => {
  const state = run([
    ['agent_spawn', { name: 'boss', isMain: true }],
    ['agent_spawn', { name: 'team #2', parent: 'boss' }],
    ['graph_structure', { ...REACT_STRUCTURE, agent: 'team #2' }],
    ['node_start', { agent: 'team #2', node: 'agent', step: 1, from: ['__start__'] }],
  ])
  assert.equal(state.graphs.get('team #2')!.parent, 'boss')
  // The canvas drops finished agents; the graph keeps the link
  state.agents.delete('team #2')
  const after = processEvent({ time: 9, type: 'node_end', payload: { agent: 'team #2', node: 'agent', step: 1 } }, state, ctx)
  assert.equal(after.graphs.get('team #2')!.parent, 'boss')
})

test('a loop leaving a left-hand node is routed on the left, clear of its siblings', () => {
  // Swarm-like: architect hands to security or performance; security hands back
  const layout = layoutGraph(graphOf({
    agent: 's',
    nodes: ['__start__', 'architect', 'security', 'performance', '__end__'].map(id => ({ id, label: id, kind: id.startsWith('__') ? (id === '__start__' ? 'start' : 'end') : 'node' })),
    edges: [['__start__', 'architect'], ['architect', 'security'], ['architect', 'performance'], ['security', 'architect'], ['performance', '__end__']]
      .map(([source, target]) => ({ source, target, conditional: false })),
  }))
  const security = layout.nodes.get('security')!, performance = layout.nodes.get('performance')!
  assert.ok(security.x < performance.x)
  const loop = layout.edges.get('security->architect')!
  assert.equal(loop.kind, 'back')
  assert.ok(loop.labelX < security.x, 'loop lane is left of the security node')
  assert.ok(Math.min(...[...layout.nodes.values()].map(n => n.x)) > 0, 'room was made for the left lane')
})
