import { test } from 'node:test'
import { strict as assert } from 'node:assert'

import { detectStateChanges } from '../web/components/agent-visualizer/canvas/detect-state-changes'
import { isGuardrail } from '../web/components/agent-visualizer/canvas/draw-tool-calls'
import { modelUsage } from '../web/components/agent-visualizer/model-legend'
import { processEvent, type ProcessEventContext } from '../web/hooks/simulation/process-event'
import { createEmptyState, type SimulationState } from '../web/hooks/simulation/types'
import { modelColor } from '../web/lib/model-colors'
import type { Agent, SimulationEvent, ToolCallNode } from '../web/lib/agent-types'

const ctx: ProcessEventContext = {
  syncForceSimulation() {},
  findToolSlot: () => ({ x: 0, y: 0 }),
  getContextWindowSize: () => 200_000,
  blockIdCounter: { current: 0 },
  skipForceSync: true,
}

function run(rows: Array<[number, string, Record<string, unknown>]>): SimulationState {
  let state = createEmptyState()
  for (const [time, type, payload] of rows) {
    state = processEvent({ time, type, payload } as SimulationEvent, { ...state, currentTime: time }, ctx)
  }
  return state
}

function agent(overrides: Partial<Agent> = {}): Agent {
  return {
    id: 'a', name: 'a', state: 'thinking', parentId: null, tokensUsed: 0, tokensMax: 100_000,
    contextBreakdown: { systemPrompt: 0, userMessages: 0, toolResults: 0, reasoning: 0, subagentResults: 0 },
    toolCalls: 0, timeAlive: 0, x: 10, y: 20, vx: 0, vy: 0, pinned: false, isMain: false,
    spawnTime: 0, opacity: 1, scale: 1, messageBubbles: [], ...overrides,
  } as Agent
}

test('a sharp drop in context is shown as compaction', () => {
  const before = new Map([['a', 80_000]])
  const states = new Map([['a', 'thinking']])
  const compacted = detectStateChanges(new Map([['a', agent({ tokensUsed: 20_000 })]]), new Map(), states, new Map(), before)
  assert.deepEqual(compacted.transitions, [{ kind: 'context_compacted' }])
  assert.equal(compacted.effects[0].type, 'compact')
  assert.equal(compacted.effects[0].label, 'context 80k → 20k')
  assert.equal(compacted.newTokens.get('a'), 20_000)

  // Growing, a small dip, or a drop from a tiny context are not compaction
  for (const [was, now] of [[20_000, 30_000], [80_000, 70_000], [5_000, 1_000]]) {
    const r = detectStateChanges(new Map([['a', agent({ tokensUsed: now })]]), new Map(), states, new Map(), new Map([['a', was]]))
    assert.equal(r.effects.length, 0, `${was} → ${now}`)
  }
})

test('a tool error ripples from the card and its agent', () => {
  const tool = { id: 't', agentId: 'a', toolName: 'fetch', state: 'error', args: '', x: 100, y: 50, startTime: 0, opacity: 1 } as ToolCallNode
  const r = detectStateChanges(new Map([['a', agent()]]), new Map([['t', tool]]), new Map([['a', 'thinking']]), new Map([['t', 'running']]))
  assert.deepEqual(r.transitions, [{ kind: 'tool_error' }])
  assert.deepEqual(r.effects.map(e => [e.type, e.x, e.y]), [['error_ripple', 100, 50], ['error_ripple', 10, 20]])
})

test('a call after the same tool failed is recorded as a retry', () => {
  const state = run([
    [0, 'agent_spawn', { name: 'worker', isMain: true }],
    [1, 'tool_call_start', { agent: 'worker', tool: 'fetch', args: 'a' }],
    [2, 'tool_call_end', { agent: 'worker', tool: 'fetch', isError: true, errorMessage: 'timeout' }],
    [5, 'tool_call_start', { agent: 'worker', tool: 'fetch', args: 'a' }],
    [6, 'tool_call_end', { agent: 'worker', tool: 'fetch', isError: true, errorMessage: 'timeout' }],
    [9, 'tool_call_start', { agent: 'worker', tool: 'fetch', args: 'a' }],
    [10, 'tool_call_start', { agent: 'worker', tool: 'other' }],
  ])
  const calls = [...state.toolCalls.values()].sort((a, b) => a.startTime - b.startTime)
  assert.deepEqual(calls.map(c => [c.toolName, c.attempt ?? 1]), [['fetch', 1], ['fetch', 2], ['fetch', 3], ['other', 1]])
  assert.equal(calls[1].retryOf, calls[0].id)
  assert.equal(calls[2].retryOf, calls[1].id)
})

test('guardrail checks are recognised by name', () => {
  assert.ok(isGuardrail({ toolName: 'guardrail: pii_filter' } as ToolCallNode))
  assert.ok(!isGuardrail({ toolName: 'search_guardrails_docs' } as ToolCallNode))
})

test('model colors are stable, by family', () => {
  assert.equal(modelColor(undefined), undefined)
  assert.equal(modelColor('claude-opus-4-1'), modelColor('demo-opus-4-6'))
  assert.notEqual(modelColor('claude-opus-4-1'), modelColor('claude-sonnet-4-5'))
  assert.notEqual(modelColor('gpt-5'), modelColor('gpt-5-mini'))
  assert.equal(modelColor('some-new-model'), modelColor('some-new-model'))
})

test('model usage totals the whole run, finished agents included', () => {
  const events = [
    [0, 'agent_spawn', { name: 'lead', isMain: true }],
    [0, 'model_detected', { agent: 'lead', model: 'claude-opus-4-1' }],
    [1, 'context_update', { agent: 'lead', tokens: 10_000 }],
    [1, 'agent_spawn', { name: 'w1', parent: 'lead' }],
    [1, 'model_detected', { agent: 'w1', model: 'claude-haiku-4-5' }],
    [2, 'context_update', { agent: 'w1', tokens: 4_000 }],
    [3, 'agent_complete', { name: 'w1' }],
    [3, 'agent_spawn', { name: 'w2', parent: 'lead' }],
    [3, 'model_detected', { agent: 'w2', model: 'claude-haiku-4-5' }],
    [4, 'context_update', { agent: 'w2', tokens: 6_000 }],
    [5, 'agent_complete', { name: 'w2' }],
  ].map(([time, type, payload]) => ({ time, type, payload } as SimulationEvent))
  const usage = modelUsage(events, 5)
  assert.deepEqual(usage.map(u => [u.model, u.agents, u.tokens]), [['claude-opus-4-1', 1, 10_000], ['claude-haiku-4-5', 2, 10_000]])
  assert.ok(usage[0].cost > usage[1].cost, 'Opus costs more per token')
  assert.deepEqual(modelUsage(events, 2).map(u => [u.model, u.tokens]), [['claude-opus-4-1', 10_000], ['claude-haiku-4-5', 4_000]])
})
