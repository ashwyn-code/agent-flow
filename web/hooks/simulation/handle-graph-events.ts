import { graphEdgeId, type AgentGraph, type GraphNodeInfo, type GraphNodeKind } from '@/lib/agent-types'
import type { MutableEventState } from './process-event'
import { asString } from './types'

const NODE_KINDS: ReadonlySet<string> = new Set(['start', 'end', 'node', 'subgraph'])

function emptyGraph(agent: string, parent: string | null): AgentGraph {
  return { agent, parent, nodes: {}, edges: {}, order: [], hasStructure: false, lastStep: 0, totalHops: 0 }
}

function defaultKind(id: string): GraphNodeKind {
  return id === '__start__' ? 'start' : id === '__end__' ? 'end' : 'node'
}

function newNode(id: string, kind: GraphNodeKind, label: string, declared: boolean): GraphNodeInfo {
  return { id, label, kind, visits: 0, running: 0, declared }
}

/** Copy-on-write access to one agent's graph within a processEvent call. */
function editGraph(state: MutableEventState, agent: string): AgentGraph {
  const prev = state.graphs.get(agent) ?? emptyGraph(agent, state.agents.get(agent)?.parentId ?? null)
  const next: AgentGraph = {
    ...prev,
    parent: prev.parent ?? state.agents.get(agent)?.parentId ?? null,
    nodes: { ...prev.nodes }, edges: { ...prev.edges }, order: [...prev.order],
  }
  state.graphs.set(agent, next)
  return next
}

function ensureNode(graph: AgentGraph, id: string): GraphNodeInfo {
  const existing = graph.nodes[id]
  if (existing) return existing
  const node = newNode(id, defaultKind(id), id, false)
  graph.nodes[id] = node
  graph.order.push(id)
  return node
}

export function handleGraphStructure(payload: Record<string, unknown>, state: MutableEventState): void {
  const agent = asString(payload.agent)
  if (!agent) return
  const graph = editGraph(state, agent)
  graph.hasStructure = true

  const nodes = Array.isArray(payload.nodes) ? payload.nodes : []
  for (const raw of nodes) {
    if (!raw || typeof raw !== 'object') continue
    const r = raw as Record<string, unknown>
    const id = asString(r.id)
    if (!id) continue
    const kind = typeof r.kind === 'string' && NODE_KINDS.has(r.kind) ? r.kind as GraphNodeKind : defaultKind(id)
    const label = asString(r.label, id) || id
    const existing = graph.nodes[id]
    if (existing) {
      graph.nodes[id] = { ...existing, kind, label, declared: true }
    } else {
      graph.nodes[id] = newNode(id, kind, label, true)
      graph.order.push(id)
    }
  }

  const edges = Array.isArray(payload.edges) ? payload.edges : []
  for (const raw of edges) {
    if (!raw || typeof raw !== 'object') continue
    const r = raw as Record<string, unknown>
    const source = asString(r.source)
    const target = asString(r.target)
    if (!source || !target) continue
    ensureNode(graph, source)
    ensureNode(graph, target)
    const id = graphEdgeId(source, target)
    const existing = graph.edges[id]
    graph.edges[id] = {
      id, source, target,
      conditional: r.conditional === true,
      label: typeof r.label === 'string' ? r.label : undefined,
      traversals: existing?.traversals ?? 0,
      lastTime: existing?.lastTime,
      declared: true,
    }
  }
}

export function handleNodeStart(payload: Record<string, unknown>, currentTime: number, state: MutableEventState): void {
  const agent = asString(payload.agent)
  const nodeId = asString(payload.node)
  if (!agent || !nodeId) return
  const graph = editGraph(state, agent)
  const step = typeof payload.step === 'number' ? payload.step : graph.lastStep + 1
  graph.lastStep = Math.max(graph.lastStep, step)

  const sources = Array.isArray(payload.from) ? payload.from.filter((s): s is string => typeof s === 'string' && s.length > 0) : []
  for (const source of sources) {
    const src = ensureNode(graph, source)
    // __start__ never gets its own node_start; count it as visited when left.
    if (src.kind === 'start') graph.nodes[source] = { ...src, visits: Math.max(src.visits, 1), lastVisitTime: currentTime }
  }

  const node = ensureNode(graph, nodeId)
  const isEnd = node.kind === 'end'
  graph.nodes[nodeId] = {
    ...node,
    visits: node.visits + 1,
    running: isEnd ? 0 : node.running + 1,
    lastStep: step,
    lastVisitTime: currentTime,
    error: undefined,
  }

  for (const source of sources) {
    const id = graphEdgeId(source, nodeId)
    const edge = graph.edges[id]
    graph.edges[id] = edge
      ? { ...edge, traversals: edge.traversals + 1, lastTime: currentTime }
      : { id, source, target: nodeId, conditional: false, traversals: 1, lastTime: currentTime, declared: false }
    graph.totalHops++
  }
}

export function handleNodeEnd(payload: Record<string, unknown>, state: MutableEventState): void {
  const agent = asString(payload.agent)
  const nodeId = asString(payload.node)
  const existing = agent ? state.graphs.get(agent)?.nodes[nodeId] : undefined
  if (!existing) return
  const graph = editGraph(state, agent)
  graph.nodes[nodeId] = {
    ...existing,
    running: Math.max(0, existing.running - 1),
    error: typeof payload.error === 'string' ? payload.error : existing.error,
  }
}

/** A finished agent has nothing in flight, even if a node_end was lost. */
export function handleGraphAgentComplete(payload: Record<string, unknown>, state: MutableEventState): void {
  const agent = asString(payload.name)
  const prev = state.graphs.get(agent)
  if (!prev || !Object.values(prev.nodes).some(n => n.running > 0)) return
  const graph = editGraph(state, agent)
  for (const [id, node] of Object.entries(graph.nodes)) {
    if (node.running > 0) graph.nodes[id] = { ...node, running: 0 }
  }
}
