/**
 * Two runs of the same graph, side by side: which routes and nodes are new
 * or gone, how each node's time, tokens, cost and errors changed, and how
 * the runs compare overall. Used by the Graph panel's Compare view.
 */
import { graphEdgeId, type AgentGraph, type SimulationEvent } from './agent-types'
import { aggregateGraph, baseName } from './multi-run'
import { analyzeRun, type AnalyzeOptions, type NodeMetrics } from './run-analysis'

export type DiffStatus = 'both' | 'added' | 'removed'

export interface NodeDiff {
  status: DiffStatus
  current?: NodeMetrics
  baseline?: NodeMetrics
  /** current − baseline (a node missing from a run counts as zero) */
  time: number
  tokens: number
  cost: number
  errors: number
}

export interface RunTotals {
  duration: number
  agents: number
  toolCalls: number
  errors: number
  tokens: number
  cost: number
}

export interface RunDiff {
  name: string
  /** Union of both runs' graphs */
  graph: AgentGraph
  edges: Map<string, DiffStatus>
  nodes: Map<string, NodeDiff>
  current: RunTotals
  baseline: RunTotals
}

/** The graph instance in a session: the exact name, else the first with the same base name. */
function instanceIn(events: readonly SimulationEvent[], graphAgent: string): string | null {
  const base = baseName(graphAgent)
  let first: string | null = null
  for (const e of events) {
    if (e.type !== 'graph_structure' && e.type !== 'node_start') continue
    const agent = (e.payload as Record<string, unknown>).agent
    if (agent === graphAgent) return graphAgent
    if (first === null && typeof agent === 'string' && baseName(agent) === base) first = agent
  }
  return first
}

function edgesTaken(events: readonly SimulationEvent[], instance: string): Set<string> {
  const taken = new Set<string>()
  for (const e of events) {
    const p = e.payload as Record<string, unknown>
    if (e.type !== 'node_start' || p.agent !== instance || typeof p.node !== 'string') continue
    for (const source of Array.isArray(p.from) ? p.from : []) if (typeof source === 'string') taken.add(graphEdgeId(source, p.node))
  }
  return taken
}

function totals(events: readonly SimulationEvent[], options: AnalyzeOptions): RunTotals {
  const a = analyzeRun(events, Infinity, options)
  const main = a.lanes.find(l => l.isMain) ?? a.lanes[0]
  const duration = main && main.intervals.length
    ? Math.max(...main.intervals.map(i => i.end ?? a.end)) - main.intervals[0].start
    : a.end - a.start
  const costOf = options.costOf ?? (() => 0)
  let toolCalls = 0, errors = 0, tokens = 0, cost = 0
  for (const lane of a.lanes) {
    toolCalls += lane.tools.length
    errors += lane.errors
    tokens += lane.tokens
    cost += costOf(lane.tokens, lane.model)
  }
  return { duration, agents: a.lanes.length, toolCalls, errors, tokens, cost }
}

const ZERO: NodeMetrics = { runs: 0, time: 0, tokens: 0, cost: 0, errors: 0 }

/** Compare `current` with `baseline` for the graph named like `graphAgent`.
 *  Returns null when either run doesn't have that graph. */
export function diffRuns(
  current: readonly SimulationEvent[],
  baseline: readonly SimulationEvent[],
  graphAgent: string,
  options: AnalyzeOptions = {},
): RunDiff | null {
  const cur = instanceIn(current, graphAgent)
  const base = instanceIn(baseline, graphAgent)
  if (!cur || !base) return null
  const union = aggregateGraph([current.filter(e => !isOtherInstance(e, cur, graphAgent)), baseline.filter(e => !isOtherInstance(e, base, graphAgent))], graphAgent, options)
  if (!union) return null

  const curEdges = edgesTaken(current, cur)
  const baseEdges = edgesTaken(baseline, base)
  const edges = new Map<string, DiffStatus>()
  for (const id of curEdges) edges.set(id, baseEdges.has(id) ? 'both' : 'added')
  for (const id of baseEdges) if (!curEdges.has(id)) edges.set(id, 'removed')

  const curMetrics = analyzeRun(current, Infinity, options).nodeMetrics.get(cur) ?? new Map()
  const baseMetrics = analyzeRun(baseline, Infinity, options).nodeMetrics.get(base) ?? new Map()
  const nodes = new Map<string, NodeDiff>()
  for (const id of new Set([...curMetrics.keys(), ...baseMetrics.keys()])) {
    const c = curMetrics.get(id)
    const b = baseMetrics.get(id)
    const cm = c ?? ZERO
    const bm = b ?? ZERO
    nodes.set(id, {
      status: c && b ? 'both' : c ? 'added' : 'removed',
      ...(c ? { current: c } : {}),
      ...(b ? { baseline: b } : {}),
      time: cm.time - bm.time,
      tokens: cm.tokens - bm.tokens,
      cost: cm.cost - bm.cost,
      errors: cm.errors - bm.errors,
    })
  }

  return { name: union.name, graph: union.graph, edges, nodes, current: totals(current, options), baseline: totals(baseline, options) }
}

/** Events of other instances of the same graph in one session (e.g. its
 *  parallel siblings), which the comparison leaves out. */
function isOtherInstance(e: SimulationEvent, instance: string, graphAgent: string): boolean {
  const agent = (e.payload as Record<string, unknown>).agent
  return (e.type === 'graph_structure' || e.type === 'node_start' || e.type === 'node_end')
    && typeof agent === 'string' && agent !== instance && baseName(agent) === baseName(graphAgent)
}

/** Sessions that ran the graph named like `graphAgent`, other than `exclude`. */
export function sessionsWithGraph(sessions: ReadonlyMap<string, readonly SimulationEvent[]>, graphAgent: string, exclude: string | null): string[] {
  const out: string[] = []
  for (const [id, events] of sessions) if (id !== exclude && instanceIn(events, graphAgent)) out.push(id)
  return out
}
