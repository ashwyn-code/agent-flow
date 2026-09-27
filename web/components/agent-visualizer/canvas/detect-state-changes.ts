import type { Agent, ToolCallNode } from '@/lib/agent-types'
import { FX, NODE } from '@/lib/agent-types'
import { COLORS } from '@/lib/colors'
import { CONTEXT_RING } from '@/lib/canvas-constants'
import { formatTokens } from '@/lib/utils'
import type { VisualEffect } from './draw-effects'

/** Context that shrinks by at least this share (from a meaningful size) was
 *  compacted or truncated. */
const COMPACTION_DROP = 0.3
const COMPACTION_MIN_USAGE = 0.15

/** A semantic state transition detected between frames. */
export type StateTransition =
  | { kind: 'agent_spawn' }
  | { kind: 'agent_complete' }
  | { kind: 'tool_start' }
  | { kind: 'tool_complete' }
  | { kind: 'tool_error' }
  | { kind: 'context_compacted' }

/**
 * Compare previous and current agent/tool states and return both visual effects
 * and semantic transitions.
 *
 * This is a pure function: it reads the previous state maps, computes results,
 * and returns the effects, transitions, and updated state maps.
 *
 * Both the canvas (for visuals) and the audio system (for sounds) consume
 * these results, keeping detection logic in a single place.
 */
export function detectStateChanges(
  agents: Map<string, Agent>,
  toolCalls: Map<string, ToolCallNode>,
  prevAgentStates: Map<string, string>,
  prevToolStates: Map<string, string>,
  prevTokens: Map<string, number> = new Map(),
): {
  effects: VisualEffect[]
  transitions: StateTransition[]
  newAgentStates: Map<string, string>
  newToolStates: Map<string, string>
  newTokens: Map<string, number>
} {
  const effects: VisualEffect[] = []
  const transitions: StateTransition[] = []
  const newAgentStates = new Map<string, string>()
  const newToolStates = new Map<string, string>()
  const newTokens = new Map<string, number>()

  for (const [id, agent] of agents) {
    newAgentStates.set(id, agent.state)
    newTokens.set(id, agent.tokensUsed)
    const oldState = prevAgentStates.get(id)

    // Context compacted: tokens fell sharply from a sizeable context
    const before = prevTokens.get(id)
    if (before !== undefined && agent.tokensMax > 0 && before / agent.tokensMax >= COMPACTION_MIN_USAGE
      && agent.tokensUsed < before * (1 - COMPACTION_DROP)) {
      transitions.push({ kind: 'context_compacted' })
      const r = agent.isMain ? NODE.radiusMain : NODE.radiusSub
      effects.push({
        type: 'compact', x: agent.x, y: agent.y,
        color: COLORS.contextReasoning, age: 0, duration: FX.compactDuration,
        radius: r + CONTEXT_RING.ringOffset + 4,
        label: `context ${formatTokens(before)} → ${formatTokens(agent.tokensUsed)}`,
      })
    }

    // Spawn: new agent (wasn't in prev)
    if (!oldState) {
      transitions.push({ kind: 'agent_spawn' })
      if (agent.opacity < 0.5) {
        effects.push({
          type: 'spawn', x: agent.x, y: agent.y,
          color: COLORS.holoBase, age: 0, duration: FX.spawnDuration,
        })
      }
    }

    // Complete: just became complete
    if (oldState && oldState !== 'complete' && agent.state === 'complete') {
      transitions.push({ kind: 'agent_complete' })
      effects.push({
        type: 'complete', x: agent.x, y: agent.y,
        color: COLORS.complete, age: 0, duration: FX.completeDuration,
      })
    }
  }

  for (const [id, tool] of toolCalls) {
    newToolStates.set(id, tool.state)
    const oldState = prevToolStates.get(id)

    // Tool just started running
    if (!oldState && tool.state === 'running') {
      transitions.push({ kind: 'tool_start' })
    }

    // Tool just completed
    if (oldState === 'running' && tool.state === 'complete') {
      transitions.push({ kind: 'tool_complete' })
      const particleData: VisualEffect['particles'] = []
      for (let i = 0; i < FX.shatterCount; i++) {
        particleData.push({
          angle: (i / FX.shatterCount) * Math.PI * 2 + Math.random() * 0.5,
          speed: FX.shatterSpeed.min + Math.random() * FX.shatterSpeed.range,
          size: FX.shatterSize.min + Math.random() * FX.shatterSize.range,
        })
      }
      effects.push({
        type: 'shatter', x: tool.x, y: tool.y,
        color: COLORS.return, age: 0, duration: FX.shatterDuration,
        particles: particleData,
      })
    }

    // Tool errored: shockwaves at the card and at its agent
    if (oldState === 'running' && tool.state === 'error') {
      transitions.push({ kind: 'tool_error' })
      effects.push({ type: 'error_ripple', x: tool.x, y: tool.y, color: COLORS.error, age: 0, duration: FX.errorRippleDuration, radius: 16 })
      const agent = agents.get(tool.agentId)
      if (agent) {
        effects.push({
          type: 'error_ripple', x: agent.x, y: agent.y, color: COLORS.error, age: 0, duration: FX.errorRippleDuration,
          radius: agent.isMain ? NODE.radiusMain : NODE.radiusSub,
        })
      }
    }
  }

  return { effects, transitions, newAgentStates, newToolStates, newTokens }
}
