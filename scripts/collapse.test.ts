import { test } from 'node:test'
import { strict as assert } from 'node:assert'

import { applyCollapse, collapseAllTargets, collapsedAncestors, hiddenAgents } from '../web/lib/collapse'
import { computeMinimapFrame, minimapToWorld, shouldShowMinimap, MINIMAP } from '../web/components/agent-visualizer/canvas/draw-minimap'
import type { Agent, Discovery, Edge, Particle, ToolCallNode } from '../web/lib/agent-types'

function agent(id: string, parentId: string | null, state: Agent['state'] = 'thinking', x = 0, y = 0): Agent {
  return {
    id, name: id, state, parentId, tokensUsed: 0, tokensMax: 1, contextBreakdown: { systemPrompt: 0, userMessages: 0, toolResults: 0, reasoning: 0, subagentResults: 0 },
    toolCalls: 0, timeAlive: 0, x, y, vx: 0, vy: 0, pinned: false, isMain: parentId === null, spawnTime: 0, opacity: 1, scale: 1, messageBubbles: [],
  } as Agent
}
const tool = (id: string, agentId: string, state: ToolCallNode['state'] = 'complete') =>
  ({ id, agentId, toolName: 't', state, args: '', x: 0, y: 0, startTime: 0, opacity: 1 }) as ToolCallNode

//   main ─┬─ lead ─┬─ w1 ── deep
//         │        └─ w2
//         └─ solo
const agents = new Map([
  ['main', agent('main', null)],
  ['lead', agent('lead', 'main', 'idle')],
  ['w1', agent('w1', 'lead', 'tool_calling')],
  ['deep', agent('deep', 'w1', 'error')],
  ['w2', agent('w2', 'lead', 'complete')],
  ['solo', agent('solo', 'main')],
])
const toolCalls = new Map([['t1', tool('t1', 'w2', 'error')], ['t2', tool('t2', 'solo')]])
const edges: Edge[] = [
  { id: 'e-main-lead', from: 'main', to: 'lead', type: 'parent-child', opacity: 1 },
  { id: 'e-lead-w1', from: 'lead', to: 'w1', type: 'parent-child', opacity: 1 },
  { id: 'e-w2-t1', from: 'w2', to: 't1', type: 'tool', opacity: 1 },
  { id: 'e-solo-t2', from: 'solo', to: 't2', type: 'tool', opacity: 1 },
]
const particles = [{ id: 'p1', edgeId: 'e-lead-w1' }, { id: 'p2', edgeId: 'e-solo-t2' }] as Particle[]
const discoveries = [{ id: 'd1', agentId: 'deep' }, { id: 'd2', agentId: 'main' }] as Discovery[]
const scene = { agents, toolCalls, edges, particles, discoveries }

test('collapsing hides the whole subtree, with what it hid rolled up', () => {
  const visible = applyCollapse(scene, new Set(['lead']))
  assert.deepEqual([...visible.agents.keys()], ['main', 'lead', 'solo'])
  assert.deepEqual([...visible.toolCalls.keys()], ['t2'])
  assert.deepEqual(visible.edges.map(e => e.id), ['e-main-lead', 'e-solo-t2'])
  assert.deepEqual(visible.particles.map(p => p.id), ['p2'])
  assert.deepEqual(visible.discoveries.map(d => d.id), ['d2'])
  assert.deepEqual(visible.collapsed.get('lead'), { hidden: 3, active: true, error: true })
})

test('a collapsed agent inside another collapsed subtree rolls up into the outer one', () => {
  const visible = applyCollapse(scene, new Set(['lead', 'w1']))
  assert.deepEqual([...visible.collapsed.keys()], ['lead'])
  assert.equal(visible.collapsed.get('lead')!.hidden, 3)
})

test('nothing collapsed: the scene passes through', () => {
  const visible = applyCollapse(scene, new Set())
  assert.equal(visible.agents, agents)
  assert.equal(visible.edges, edges)
  assert.equal(visible.collapsed.size, 0)
  // A collapsed leaf hides nothing but is still marked
  assert.deepEqual(applyCollapse(scene, new Set(['solo'])).collapsed.get('solo'), { hidden: 0, active: false, error: false })
})

test('helpers: hidden set, ancestors to unfold, collapse-all targets', () => {
  assert.deepEqual([...hiddenAgents(agents, new Set(['w1']))], ['deep'])
  assert.deepEqual(collapsedAncestors(agents, 'deep', new Set(['lead', 'w1'])), ['w1', 'lead'])
  assert.deepEqual(collapsedAncestors(agents, 'solo', new Set(['lead'])), [])
  assert.deepEqual(collapseAllTargets(agents).sort(), ['lead', 'w1'])
})

test('minimap: shown for bigger runs or off-screen agents; clicks map back to the world', () => {
  const view = { x: 400, y: 300, scale: 1 }   // world (0,0) at the screen center of 800x600
  const few = new Map([['a', agent('a', null, 'thinking', 0, 0)], ['b', agent('b', 'a', 'thinking', 100, 0)]])
  assert.equal(shouldShowMinimap(few, view, 800, 600), false)
  const offscreen = new Map([...few, ['c', agent('c', 'a', 'thinking', 2000, 0)]])
  assert.equal(shouldShowMinimap(offscreen, view, 800, 600), true)
  const many = new Map(Array.from({ length: MINIMAP.minAgents }, (_, i) => [`a${i}`, agent(`a${i}`, null, 'idle', i, 0)] as const))
  assert.equal(shouldShowMinimap(many, view, 800, 600), true)

  const frame = computeMinimapFrame(offscreen, new Map(), view, 800, 600)
  // The minimap's center column maps back into the world bounds, and round-trips
  const px = frame.offsetX + (1000 - frame.minX) * frame.scale
  const py = frame.offsetY + (0 - frame.minY) * frame.scale
  const world = minimapToWorld(frame, px, py)
  assert.ok(Math.abs(world.x - 1000) < 1e-6 && Math.abs(world.y) < 1e-6)
  assert.ok(frame.offsetX >= MINIMAP.padding - 1e-9 && frame.offsetY >= MINIMAP.padding - 1e-9)
})
