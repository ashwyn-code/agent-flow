/**
 * Collapsing subtrees on the canvas: the descendants of a collapsed agent
 * (their tool calls, edges, particles and discoveries too) are left out of
 * what gets drawn, fitted and hit-tested, and the collapsed agent carries a
 * summary of what's hidden under it.
 */
import type { Agent, Discovery, Edge, Particle, ToolCallNode } from './agent-types'

export interface CollapsedSummary {
  /** Agents hidden under this one */
  hidden: number
  /** Something hidden is still working */
  active: boolean
  /** Something hidden failed (an agent in error, or a failed tool call) */
  error: boolean
}

export interface SceneParts {
  agents: Map<string, Agent>
  toolCalls: Map<string, ToolCallNode>
  edges: Edge[]
  particles: Particle[]
  discoveries: Discovery[]
}

export interface VisibleScene extends SceneParts {
  /** Collapsed agents that are themselves visible, with what they hide */
  collapsed: Map<string, CollapsedSummary>
}

const ACTIVE = new Set(['thinking', 'tool_calling', 'waiting_permission'])

function childrenOf(agents: Map<string, Agent>): Map<string, string[]> {
  const children = new Map<string, string[]>()
  for (const [id, agent] of agents) {
    if (!agent.parentId || agent.parentId === id) continue
    const list = children.get(agent.parentId) ?? []
    list.push(id)
    children.set(agent.parentId, list)
  }
  return children
}

/** Agents with at least one subagent on the canvas */
export function agentsWithChildren(agents: Map<string, Agent>): Set<string> {
  return new Set(childrenOf(agents).keys())
}

/** Ids of the agents that are hidden because an ancestor is collapsed. */
export function hiddenAgents(agents: Map<string, Agent>, collapsed: ReadonlySet<string>): Set<string> {
  const hidden = new Set<string>()
  if (!collapsed.size) return hidden
  const children = childrenOf(agents)
  const stack = [...collapsed].filter(id => agents.has(id))
  while (stack.length) {
    const id = stack.pop()!
    for (const child of children.get(id) ?? []) {
      if (!hidden.has(child)) {
        hidden.add(child)
        stack.push(child)
      }
    }
  }
  return hidden
}

/** Collapsed ancestors of an agent (to expand when it's selected elsewhere). */
export function collapsedAncestors(agents: Map<string, Agent>, id: string, collapsed: ReadonlySet<string>): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (let p = agents.get(id)?.parentId ?? null; p && !seen.has(p); p = agents.get(p)?.parentId ?? null) {
    seen.add(p)
    if (collapsed.has(p)) out.push(p)
  }
  return out
}

/** What's left to draw once collapsed subtrees are folded away. */
export function applyCollapse(scene: SceneParts, collapsed: ReadonlySet<string>): VisibleScene {
  const hidden = hiddenAgents(scene.agents, collapsed)
  if (!hidden.size) {
    return { ...scene, collapsed: new Map([...collapsed].filter(id => scene.agents.has(id)).map(id => [id, { hidden: 0, active: false, error: false }])) }
  }

  // Which visible collapsed agent each hidden agent rolls up into
  const rollup = new Map<string, string>()
  const topOf = (id: string): string | null => {
    const cached = rollup.get(id)
    if (cached) return cached
    let top: string | null = null
    const seen = new Set<string>()
    for (let p = scene.agents.get(id)?.parentId ?? null; p && !seen.has(p); p = scene.agents.get(p)?.parentId ?? null) {
      seen.add(p)
      if (collapsed.has(p) && !hidden.has(p)) { top = p; break }
    }
    if (top) rollup.set(id, top)
    return top
  }

  const summaries = new Map<string, CollapsedSummary>()
  for (const id of collapsed) {
    if (scene.agents.has(id) && !hidden.has(id)) summaries.set(id, { hidden: 0, active: false, error: false })
  }
  for (const id of hidden) {
    const top = topOf(id)
    const summary = top ? summaries.get(top) : undefined
    if (!summary) continue
    const agent = scene.agents.get(id)!
    summary.hidden++
    if (ACTIVE.has(agent.state)) summary.active = true
    if (agent.state === 'error') summary.error = true
  }

  const agents = new Map<string, Agent>()
  for (const [id, agent] of scene.agents) if (!hidden.has(id)) agents.set(id, agent)
  const toolCalls = new Map<string, ToolCallNode>()
  const hiddenTools = new Set<string>()
  for (const [id, tool] of scene.toolCalls) {
    if (hidden.has(tool.agentId)) {
      hiddenTools.add(id)
      const top = topOf(tool.agentId)
      const summary = top ? summaries.get(top) : undefined
      if (summary && tool.state === 'error') summary.error = true
      if (summary && tool.state === 'running') summary.active = true
    } else {
      toolCalls.set(id, tool)
    }
  }
  const gone = (id: string) => hidden.has(id) || hiddenTools.has(id)
  const edges = scene.edges.filter(e => !gone(e.from) && !gone(e.to))
  const edgeIds = new Set(edges.map(e => e.id))
  const particles = scene.particles.filter(p => edgeIds.has(p.edgeId))
  const discoveries = scene.discoveries.filter(d => !hidden.has(d.agentId))
  return { agents, toolCalls, edges, particles, discoveries, collapsed: summaries }
}

/** Every agent that has subagents, except the main one: leaves the main
 *  agent and its direct children on screen. */
export function collapseAllTargets(agents: Map<string, Agent>): string[] {
  const withChildren = agentsWithChildren(agents)
  return [...withChildren].filter(id => !agents.get(id)?.isMain)
}
