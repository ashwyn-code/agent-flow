/**
 * Derives timing, cost and critical-path information for a session from its
 * event log, as it stood at a given moment (so it follows scrubbing too).
 *
 * Used by the swimlane timeline (lanes, parallel tool calls, the critical
 * path) and by the Graph panel's metric overlays (time, tokens, cost and
 * errors per node).
 */
import type { SimulationEvent } from './agent-types'

/** Seconds; below this, gaps and items are treated as instantaneous */
const EPS = 1e-6

export interface ToolSpan {
  id: string
  agent: string
  tool: string
  label: string
  start: number
  end?: number
  error?: string
  /** Sub-row within the agent's lane (parallel calls stack) */
  row: number
}

export interface AgentInterval {
  start: number
  end?: number
}

export interface Lane {
  name: string
  parent: string | null
  depth: number
  isMain: boolean
  /** An agent name can run more than once (e.g. a node in a loop) */
  intervals: AgentInterval[]
  tools: ToolSpan[]
  /** Sub-rows needed for overlapping tool calls */
  rows: number
  model?: string
  /** Largest context seen */
  tokens: number
  errors: number
}

export interface CriticalSegment {
  lane: string
  /** A tool call, or the agent's own time (model calls, thinking, waiting) */
  kind: 'tool' | 'self'
  start: number
  end: number
  label: string
  toolId?: string
  /** Tool name, for tool segments */
  tool?: string
}

export interface NodeMetrics {
  runs: number
  /** Seconds spent in the node, summed over its runs */
  time: number
  /** Context tokens of the agents that ran for it (and their subagents) */
  tokens: number
  cost: number
  errors: number
}

export interface RunAnalysis {
  /** Lanes in tree order (each parent followed by its children) */
  lanes: Lane[]
  start: number
  end: number
  critical: CriticalSegment[]
  criticalToolIds: Set<string>
  criticalLanes: Set<string>
  /** Critical-path time in tool calls vs. in agents themselves */
  breakdown: { tools: number; self: number }
  /** Largest items on the critical path */
  topCritical: { lane: string; label: string; kind: 'tool' | 'self'; duration: number }[]
  /** Graph agent → node id → metrics */
  nodeMetrics: Map<string, Map<string, NodeMetrics>>
}

export interface AnalyzeOptions {
  /** $ for a number of tokens on a model */
  costOf?: (tokens: number, model?: string) => number
}

interface Instance {
  id: string
  lane: string
  parent: Instance | null
  start: number
  end?: number
  tokens: number
  model?: string
  toolErrors: number
  children: Instance[]
}

interface NodeRun {
  node: string
  start: number
  end?: number
  error?: string
  instance?: Instance
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '')

export function analyzeRun(events: readonly SimulationEvent[], now: number, options: AnalyzeOptions = {}): RunAnalysis {
  const lanes = new Map<string, Lane>()
  const open = new Map<string, Instance>()     // lane name → running instance
  const instances: Instance[] = []
  const nodeRuns = new Map<string, NodeRun[]>()
  let toolSeq = 0
  let first = Infinity
  let last = -Infinity

  const lane = (name: string, parent: string | null = null): Lane => {
    let l = lanes.get(name)
    if (!l) {
      l = { name, parent, depth: 0, isMain: false, intervals: [], tools: [], rows: 1, tokens: 0, errors: 0 }
      lanes.set(name, l)
    }
    return l
  }

  const close = (name: string, t: number) => {
    const inst = open.get(name)
    if (!inst) return
    inst.end = t
    open.delete(name)
    const l = lanes.get(name)
    const interval = l?.intervals[l.intervals.length - 1]
    if (interval && interval.end === undefined) interval.end = t
    for (const span of l?.tools ?? []) if (span.end === undefined) span.end = t
  }

  for (const event of events) {
    const t = event.time
    if (t > now + EPS) break
    first = Math.min(first, t)
    last = Math.max(last, t)
    const p = event.payload as Record<string, unknown>
    switch (event.type) {
      case 'agent_spawn': {
        const name = str(p.name)
        if (!name || open.has(name)) break
        const parent = str(p.parent) || null
        const l = lane(name, parent)
        if (parent && !l.parent) l.parent = parent
        if (p.isMain === true) l.isMain = true
        if (typeof p.model === 'string') l.model = p.model
        l.intervals.push({ start: t })
        const parentInst = parent ? open.get(parent) ?? null : null
        const inst: Instance = { id: `${name}@${l.intervals.length}`, lane: name, parent: parentInst, start: t, tokens: 0, model: l.model, toolErrors: 0, children: [] }
        parentInst?.children.push(inst)
        instances.push(inst)
        open.set(name, inst)
        // A graph's node runs the agent spawned while it is running
        if (parent) {
          const runs = nodeRuns.get(parent)
          const run = runs && [...runs].reverse().find(r => r.end === undefined && !r.instance)
          if (run) run.instance = inst
        }
        break
      }
      case 'agent_complete': {
        const name = str(p.name)
        close(name, t)
        // Children still running end with their parent (as on the canvas)
        for (const l of lanes.values()) if (l.parent === name) close(l.name, t)
        break
      }
      case 'tool_call_start': {
        const agent = str(p.agent)
        if (!agent) break
        const tool = str(p.tool) || 'tool'
        const args = str(p.args)
        lane(agent).tools.push({
          id: `tool-${toolSeq++}`, agent, tool, start: t, row: 0,
          label: args ? `${tool}: ${args}`.slice(0, 80) : tool,
        })
        break
      }
      case 'tool_call_end': {
        const agent = str(p.agent)
        const tool = str(p.tool) || 'tool'
        const span = lanes.get(agent)?.tools.find(s => s.tool === tool && s.end === undefined)
        if (!span) break
        span.end = t
        if (p.isError === true) {
          span.error = str(p.errorMessage) || str(p.result) || 'error'
          lanes.get(agent)!.errors++
          const inst = open.get(agent)
          if (inst) inst.toolErrors++
        }
        break
      }
      case 'context_update': {
        const agent = str(p.agent)
        const tokens = typeof p.tokens === 'number' ? p.tokens : 0
        const l = lanes.get(agent)
        if (l) l.tokens = Math.max(l.tokens, tokens)
        const inst = open.get(agent)
        if (inst) inst.tokens = Math.max(inst.tokens, tokens)
        break
      }
      case 'model_detected': {
        const agent = str(p.agent)
        const model = str(p.model)
        const l = lanes.get(agent)
        if (l && model) l.model = model
        const inst = open.get(agent)
        if (inst && model) inst.model = model
        break
      }
      case 'node_start': {
        const agent = str(p.agent)
        const node = str(p.node)
        if (!agent || !node || node === '__end__' || node === '__start__') break
        const runs = nodeRuns.get(agent) ?? []
        runs.push({ node, start: t })
        nodeRuns.set(agent, runs)
        break
      }
      case 'node_end': {
        const agent = str(p.agent)
        const node = str(p.node)
        const run = nodeRuns.get(agent)?.find(r => r.node === node && r.end === undefined)
        if (run) {
          run.end = t
          if (typeof p.error === 'string' && p.error) run.error = p.error
        }
        break
      }
    }
  }

  const start = Number.isFinite(first) ? first : 0
  const end = Math.max(Number.isFinite(last) ? last : 0, now)

  // Stack overlapping tool calls into sub-rows
  for (const l of lanes.values()) {
    const rowEnds: number[] = []
    for (const span of [...l.tools].sort((a, b) => a.start - b.start)) {
      const spanEnd = span.end ?? end
      let row = rowEnds.findIndex(e => e <= span.start + EPS)
      if (row === -1) { row = rowEnds.length; rowEnds.push(spanEnd) } else rowEnds[row] = spanEnd
      span.row = row
    }
    l.rows = Math.max(1, rowEnds.length)
  }

  const ordered = treeOrder(lanes)
  const critical = criticalPath(ordered, lanes, end)
  const criticalToolIds = new Set(critical.filter(s => s.toolId).map(s => s.toolId!))
  const criticalLanes = new Set(critical.map(s => s.lane))
  const breakdown = { tools: 0, self: 0 }
  const totals = new Map<string, { lane: string; label: string; kind: 'tool' | 'self'; duration: number }>()
  for (const s of critical) {
    const d = s.end - s.start
    if (s.kind === 'tool') breakdown.tools += d
    else breakdown.self += d
    const label = s.kind === 'tool' ? s.tool ?? s.label : s.lane
    const key = `${s.kind}\u0000${s.lane}\u0000${label}`
    const entry = totals.get(key) ?? { lane: s.lane, label, kind: s.kind, duration: 0 }
    entry.duration += d
    totals.set(key, entry)
  }
  const topCritical = [...totals.values()].sort((a, b) => b.duration - a.duration).slice(0, 4)

  return {
    lanes: ordered, start, end, critical, criticalToolIds, criticalLanes, breakdown, topCritical,
    nodeMetrics: nodeMetrics(nodeRuns, end, options),
  }
}

function treeOrder(lanes: Map<string, Lane>): Lane[] {
  const children = new Map<string | null, Lane[]>()
  for (const l of lanes.values()) {
    const parent = l.parent && lanes.has(l.parent) && l.parent !== l.name ? l.parent : null
    const list = children.get(parent) ?? []
    list.push(l)
    children.set(parent, list)
  }
  const firstStart = (l: Lane) => l.intervals[0]?.start ?? l.tools[0]?.start ?? Infinity
  const out: Lane[] = []
  const seen = new Set<string>()
  const visit = (l: Lane, depth: number) => {
    if (seen.has(l.name)) return
    seen.add(l.name)
    l.depth = depth
    out.push(l)
    for (const c of (children.get(l.name) ?? []).sort((a, b) => firstStart(a) - firstStart(b))) visit(c, depth + 1)
  }
  const roots = (children.get(null) ?? []).sort((a, b) => Number(b.isMain) - Number(a.isMain) || firstStart(a) - firstStart(b))
  for (const r of roots) visit(r, 0)
  for (const l of lanes.values()) visit(l, 0)   // anything left in a parent cycle
  return out
}

interface Item { start: number; end: number; tool?: ToolSpan; child?: Lane }

/**
 * The chain of work that set the run's duration: from the main agent's end,
 * repeatedly take the item (tool call or subagent run) that finished last
 * before the current point, descend into subagents, and count the gaps as the
 * agent's own time.
 */
function criticalPath(ordered: Lane[], lanes: Map<string, Lane>, end: number): CriticalSegment[] {
  const root = ordered.find(l => l.isMain) ?? ordered[0]
  if (!root || !root.intervals.length) return []
  const childrenOf = new Map<string, Lane[]>()
  for (const l of ordered) {
    if (l.parent && lanes.has(l.parent)) {
      const list = childrenOf.get(l.parent) ?? []
      list.push(l)
      childrenOf.set(l.parent, list)
    }
  }
  const itemsOf = (l: Lane): Item[] => [
    ...l.tools.map(tool => ({ start: tool.start, end: tool.end ?? end, tool })),
    ...(childrenOf.get(l.name) ?? []).flatMap(child => child.intervals.map(i => ({ start: i.start, end: i.end ?? end, child }))),
  ]

  const out: CriticalSegment[] = []
  /** Walk back from `to` to `from` through `items`; time not covered by an
   *  item counts as `gap` (the agent's own time, or the tool it's inside). */
  const walk = (l: Lane, from: number, to: number, depth: number, items: Item[], gap: { kind: 'tool' | 'self'; label: string; toolId?: string; tool?: string }) => {
    let t = to
    const pushGap = (a: number, b: number) => {
      if (b - a > EPS) out.push({ lane: l.name, kind: gap.kind, start: a, end: b, label: gap.label, ...(gap.toolId ? { toolId: gap.toolId, tool: gap.tool } : {}) })
    }
    while (t - from > EPS) {
      let best: Item | null = null
      for (const item of items) {
        if (item.end > t + EPS || item.start >= t - EPS || item.end <= from + EPS) continue
        if (!best || item.end > best.end + EPS || (Math.abs(item.end - best.end) <= EPS && item.start < best.start)) best = item
      }
      if (!best) break
      pushGap(best.end, t)
      const s = Math.max(best.start, from)
      if (best.tool) {
        // An agent called as a tool: follow the agent inside the call
        const inner = items.filter(i => i.child && i.start >= best!.start - EPS && i.end <= best!.end + EPS)
        if (inner.length && depth < 64) {
          walk(l, s, best.end, depth + 1, inner, { kind: 'tool', label: best.tool.label, toolId: best.tool.id, tool: best.tool.tool })
        } else {
          out.push({ lane: l.name, kind: 'tool', start: s, end: best.end, label: best.tool.label, toolId: best.tool.id, tool: best.tool.tool })
        }
      } else if (best.child && depth < 64) {
        walk(best.child, s, best.end, depth + 1, itemsOf(best.child), { kind: 'self', label: best.child.name })
      }
      t = s
    }
    pushGap(from, t)
  }
  const rootStart = root.intervals[0].start
  const rootEnd = Math.max(...root.intervals.map(i => i.end ?? end))
  walk(root, rootStart, rootEnd, 0, itemsOf(root), { kind: 'self', label: root.name })
  return out.sort((a, b) => a.start - b.start)
}

function nodeMetrics(nodeRuns: Map<string, NodeRun[]>, end: number, options: AnalyzeOptions): Map<string, Map<string, NodeMetrics>> {
  const costOf = options.costOf ?? (() => 0)
  const subtree = new Map<Instance, { tokens: number; cost: number; errors: number }>()
  const sum = (inst: Instance): { tokens: number; cost: number; errors: number } => {
    const cached = subtree.get(inst)
    if (cached) return cached
    const total = { tokens: inst.tokens, cost: costOf(inst.tokens, inst.model), errors: inst.toolErrors }
    subtree.set(inst, total)   // guards against cycles
    for (const c of inst.children) {
      const s = sum(c)
      total.tokens += s.tokens
      total.cost += s.cost
      total.errors += s.errors
    }
    return total
  }

  const out = new Map<string, Map<string, NodeMetrics>>()
  for (const [agent, runs] of nodeRuns) {
    const metrics = new Map<string, NodeMetrics>()
    for (const run of runs) {
      const m = metrics.get(run.node) ?? { runs: 0, time: 0, tokens: 0, cost: 0, errors: 0 }
      m.runs++
      m.time += Math.max(0, (run.end ?? end) - run.start)
      if (run.error) m.errors++
      if (run.instance) {
        const s = sum(run.instance)
        m.tokens += s.tokens
        m.cost += s.cost
        m.errors += s.errors
      }
      metrics.set(run.node, m)
    }
    out.set(agent, metrics)
  }
  return out
}
