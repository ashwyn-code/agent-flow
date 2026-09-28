/**
 * One graph across many runs: every session the UI has seen that ran the same
 * graph (e.g. each OpenTelemetry trace of a workflow, or each adapter run)
 * counts as a run, and so does each instance of it within a session (parallel
 * subgraphs such as "research_team #2").
 *
 * Gives the union of the graph's structure, how often each edge was taken
 * (share of runs), and per-node distributions: share of runs visiting it,
 * median and p95 time, median tokens, mean cost and error rate.
 */
import { handleGraphAgentComplete, handleGraphStructure, handleNodeEnd, handleNodeStart } from '@/hooks/simulation/handle-graph-events'
import type { MutableEventState } from '@/hooks/simulation/process-event'
import { graphEdgeId, type AgentGraph, type SimulationEvent } from './agent-types'
import { analyzeRun, type AnalyzeOptions } from './run-analysis'

export interface NodeAggregate {
  /** Runs in which the node ran at least once */
  runs: number
  share: number
  timeP50: number
  timeP95: number
  tokensP50: number
  costMean: number
  /** Runs in which the node had an error */
  errorRuns: number
  errorShare: number
}

export interface EdgeAggregate {
  runs: number
  share: number
}

export interface GraphAggregate {
  /** Canonical graph name (without a "#2" suffix) */
  name: string
  /** Union of every run's structure and routes */
  graph: AgentGraph
  runs: number
  sessions: number
  nodes: Map<string, NodeAggregate>
  edges: Map<string, EdgeAggregate>
}

/** "research_team #3" → "research_team" */
export function baseName(name: string): string {
  return name.replace(/ #\d+$/, '')
}

/** Value at quantile q (0..1) of a sorted-or-not list, by nearest rank. */
export function quantile(values: number[], q: number): number {
  if (!values.length) return 0
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))]
}

function scratchState(): MutableEventState {
  return {
    agents: new Map(), toolCalls: new Map(), particles: [], edges: [], discoveries: [],
    fileAttention: new Map(), timelineEntries: new Map(), conversations: new Map(), graphs: new Map(),
  }
}

const GRAPH_EVENTS = new Set(['graph_structure', 'node_start', 'node_end'])

/**
 * Aggregate the graph named like `graphAgent` over `sessions` (each a
 * session's events). Returns null when no session ran it.
 */
export function aggregateGraph(
  sessions: Iterable<readonly SimulationEvent[]>,
  graphAgent: string,
  options: AnalyzeOptions = {},
): GraphAggregate | null {
  const name = baseName(graphAgent)
  const union = scratchState()
  let runs = 0
  let sessionCount = 0
  const perNode = new Map<string, { runs: number; times: number[]; tokens: number[]; costs: number[]; errorRuns: number }>()
  const perEdge = new Map<string, number>()

  for (const events of sessions) {
    // This session's instances of the graph
    const instances = new Set<string>()
    for (const e of events) {
      if (!GRAPH_EVENTS.has(e.type)) continue
      const agent = (e.payload as Record<string, unknown>).agent
      if (typeof agent === 'string' && baseName(agent) === name) instances.add(agent)
    }
    if (!instances.size) continue
    sessionCount++

    // Union structure and routes, with every instance folded onto one name
    for (const e of events) {
      const p = e.payload as Record<string, unknown>
      if (GRAPH_EVENTS.has(e.type) && typeof p.agent === 'string' && instances.has(p.agent)) {
        const payload = { ...p, agent: name }
        if (e.type === 'graph_structure') handleGraphStructure(payload, union)
        else if (e.type === 'node_start') handleNodeStart(payload, e.time, union)
        else handleNodeEnd(payload, union)
      }
    }
    handleGraphAgentComplete({ name }, union)

    const metrics = analyzeRun(events, Infinity, options).nodeMetrics
    for (const instance of instances) {
      runs++
      for (const [node, m] of metrics.get(instance) ?? []) {
        const entry = perNode.get(node) ?? { runs: 0, times: [], tokens: [], costs: [], errorRuns: 0 }
        entry.runs++
        entry.times.push(m.time)
        entry.tokens.push(m.tokens)
        entry.costs.push(m.cost)
        if (m.errors > 0) entry.errorRuns++
        perNode.set(node, entry)
      }
      const taken = new Set<string>()
      for (const e of events) {
        const p = e.payload as Record<string, unknown>
        if (e.type !== 'node_start' || p.agent !== instance || typeof p.node !== 'string') continue
        for (const source of Array.isArray(p.from) ? p.from : []) {
          if (typeof source === 'string') taken.add(graphEdgeId(source, p.node))
        }
      }
      for (const id of taken) perEdge.set(id, (perEdge.get(id) ?? 0) + 1)
    }
  }

  const graph = union.graphs.get(name)
  if (!graph || runs === 0) return null

  const nodes = new Map<string, NodeAggregate>()
  for (const [id, n] of perNode) {
    nodes.set(id, {
      runs: n.runs,
      share: n.runs / runs,
      timeP50: quantile(n.times, 0.5),
      timeP95: quantile(n.times, 0.95),
      tokensP50: quantile(n.tokens, 0.5),
      costMean: n.costs.reduce((a, b) => a + b, 0) / n.costs.length,
      errorRuns: n.errorRuns,
      errorShare: n.errorRuns / runs,
    })
  }
  const edges = new Map<string, EdgeAggregate>()
  for (const [id, count] of perEdge) edges.set(id, { runs: count, share: count / runs })

  // START and END are reached in every run that got there
  const start = graph.nodes.__start__
  if (start) nodes.set('__start__', { runs, share: 1, timeP50: 0, timeP95: 0, tokensP50: 0, costMean: 0, errorRuns: 0, errorShare: 0 })

  return { name, graph, runs, sessions: sessionCount, nodes, edges }
}
