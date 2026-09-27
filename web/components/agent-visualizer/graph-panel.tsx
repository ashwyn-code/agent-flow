'use client'

import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { Z, type Agent, type AgentGraph, type GraphNodeInfo, type SimulationEvent } from '@/lib/agent-types'
import { COLORS } from '@/lib/colors'
import { layoutGraph, type LayoutEdge, type LayoutNode } from '@/lib/graph-layout'
import { analyzeRun, type NodeMetrics } from '@/lib/run-analysis'
import { aggregateGraph, type GraphAggregate } from '@/lib/multi-run'
import { diffRuns, sessionsWithGraph, type DiffStatus, type RunDiff, type RunTotals } from '@/lib/run-diff'
import type { SessionInfo } from '@/lib/vscode-bridge'
import { formatTokens } from '@/lib/utils'
import { agentCost } from './canvas/draw-cost'
import { PanelHeader, SlidingPanel, stopPropagationHandlers } from './shared-ui'

/** How long (seconds) a just-taken hop keeps its flowing highlight */
const HOT_HOP_S = 1.5
// Shrinks on narrow windows (e.g. a VS Code side column)
const PANEL_W = 'min(380px, calc(100vw - 24px))'

// ─── Metric overlays ─────────────────────────────────────────────────────────

export type GraphOverlay = 'runs' | 'time' | 'tokens' | 'cost' | 'errors'

const OVERLAYS: { id: GraphOverlay; label: string; color: string; title: string }[] = [
  { id: 'runs', label: 'Runs', color: COLORS.holoBase, title: 'Which nodes ran, and how often' },
  { id: 'time', label: 'Time', color: COLORS.tool_calling, title: 'Time spent in each node, summed over its runs' },
  { id: 'tokens', label: 'Tokens', color: COLORS.contextReasoning, title: 'Context tokens used by the agents each node ran (including their subagents)' },
  { id: 'cost', label: 'Cost', color: COLORS.complete, title: 'Estimated cost of the agents each node ran' },
  { id: 'errors', label: 'Errors', color: COLORS.error, title: 'Failed tool calls and node errors' },
]

interface OverlayCell { value: number; text: string; heat: number; color: string; detail?: string }

function metricValue(m: NodeMetrics, overlay: GraphOverlay): number {
  switch (overlay) {
    case 'time': return m.time
    case 'tokens': return m.tokens
    case 'cost': return m.cost
    case 'errors': return m.errors
    default: return m.runs
  }
}

function formatMetric(value: number, overlay: GraphOverlay): string {
  switch (overlay) {
    case 'time': return value < 1 ? `${Math.round(value * 1000)}ms` : value < 60 ? `${value.toFixed(value < 10 ? 1 : 0)}s` : `${Math.floor(value / 60)}m${Math.round(value % 60)}s`
    case 'tokens': return formatTokens(value)
    case 'cost': return value < 0.01 ? `$${value.toFixed(3)}` : `$${value.toFixed(2)}`
    case 'errors': return `${value} err`
    default: return `${value}×`
  }
}

/** Per-node cells for the overlay: the value, its label, and its share of the hottest node. */
export function overlayCells(metrics: Map<string, NodeMetrics> | undefined, overlay: GraphOverlay): Map<string, OverlayCell> {
  const cells = new Map<string, OverlayCell>()
  if (!metrics || overlay === 'runs') return cells
  const color = OVERLAYS.find(o => o.id === overlay)!.color
  let max = 0
  for (const m of metrics.values()) max = Math.max(max, metricValue(m, overlay))
  for (const [id, m] of metrics) {
    const value = metricValue(m, overlay)
    // Nothing to report (e.g. a plain function node has no tokens or cost)
    if (value === 0 && overlay !== 'time') continue
    cells.set(id, { value, text: formatMetric(value, overlay), heat: max > 0 ? value / max : 0, color })
  }
  return cells
}

/** Cells for the all-runs view: what the node does across runs. */
export function aggregateCells(agg: GraphAggregate, overlay: GraphOverlay): Map<string, OverlayCell> {
  const cells = new Map<string, OverlayCell>()
  const color = OVERLAYS.find(o => o.id === overlay)!.color
  const value = (n: NonNullable<ReturnType<GraphAggregate['nodes']['get']>>) =>
    overlay === 'time' ? n.timeP95 : overlay === 'tokens' ? n.tokensP50 : overlay === 'cost' ? n.costMean : overlay === 'errors' ? n.errorShare : n.share
  let max = 0
  for (const [id, n] of agg.nodes) if (id !== '__start__' && id !== '__end__') max = Math.max(max, value(n))
  for (const [id, n] of agg.nodes) {
    if (id === '__start__' || id === '__end__') continue
    const v = value(n)
    if (v === 0 && overlay !== 'time' && overlay !== 'runs') continue
    const text = overlay === 'time' ? `${formatMetric(n.timeP50, 'time')}/${formatMetric(n.timeP95, 'time')}`
      : overlay === 'tokens' ? formatMetric(n.tokensP50, 'tokens')
      : overlay === 'cost' ? `${formatMetric(n.costMean, 'cost')}/run`
      : overlay === 'errors' ? `${Math.round(n.errorShare * 100)}% err`
      : `${Math.round(n.share * 100)}%`
    const detail = [
      `ran in ${Math.round(n.share * 100)}% of ${agg.runs} runs`,
      `time: median ${formatMetric(n.timeP50, 'time')} · p95 ${formatMetric(n.timeP95, 'time')}`,
      n.tokensP50 ? `tokens: median ${formatMetric(n.tokensP50, 'tokens')}` : null,
      n.costMean ? `cost: ${formatMetric(n.costMean, 'cost')} per run` : null,
      n.errorRuns ? `errors in ${n.errorRuns} run${n.errorRuns === 1 ? '' : 's'} (${Math.round(n.errorShare * 100)}%)` : null,
    ].filter(Boolean).join('\n')
    cells.set(id, { value: v, text, heat: max > 0 ? v / max : 0, color, detail })
  }
  return cells
}

/** Signed change, e.g. "+1.2s" / "−300ms" */
function signed(value: number, overlay: GraphOverlay): string {
  const sign = value > 0 ? '+' : value < 0 ? '\u2212' : '±'
  const abs = Math.abs(value)
  return `${sign}${overlay === 'errors' ? `${abs} err` : formatMetric(abs, overlay)}`
}

/** Cells for the Compare view: what changed at each node since the baseline. */
export function diffCells(diff: RunDiff, overlay: GraphOverlay): Map<string, OverlayCell> {
  const cells = new Map<string, OverlayCell>()
  const delta = (n: NonNullable<ReturnType<RunDiff['nodes']['get']>>) =>
    overlay === 'time' ? n.time : overlay === 'tokens' ? n.tokens : overlay === 'cost' ? n.cost : overlay === 'errors' ? n.errors : 0
  let max = 0
  for (const n of diff.nodes.values()) max = Math.max(max, Math.abs(delta(n)))
  for (const [id, n] of diff.nodes) {
    const detail = [
      n.status === 'added' ? 'only in this run' : n.status === 'removed' ? 'only in the baseline run' : 'in both runs',
      `time ${formatMetric(n.current?.time ?? 0, 'time')} (was ${formatMetric(n.baseline?.time ?? 0, 'time')})`,
      n.current?.tokens || n.baseline?.tokens ? `tokens ${formatMetric(n.current?.tokens ?? 0, 'tokens')} (was ${formatMetric(n.baseline?.tokens ?? 0, 'tokens')})` : null,
      n.current?.errors || n.baseline?.errors ? `errors ${n.current?.errors ?? 0} (was ${n.baseline?.errors ?? 0})` : null,
    ].filter(Boolean).join('\n')
    if (overlay === 'runs') {
      if (n.status === 'both') continue
      const added = n.status === 'added'
      cells.set(id, { value: 1, text: added ? 'new' : 'gone', heat: 0.35, color: added ? COLORS.complete : COLORS.error, detail })
      continue
    }
    const v = delta(n)
    // Tiny changes are noise
    if (overlay === 'time' ? Math.abs(v) < 0.01 : v === 0) continue
    // More time, tokens, cost or errors is worse
    cells.set(id, { value: Math.abs(v), text: signed(v, overlay), heat: max > 0 ? Math.abs(v) / max : 0, color: v > 0 ? COLORS.error : COLORS.complete, detail })
  }
  return cells
}

type Scope = 'run' | 'all' | 'compare'

interface GraphPanelProps {
  visible: boolean
  graphs: Map<string, AgentGraph>
  agents: Map<string, Agent>
  /** The session's events, for the metric overlays */
  events?: readonly SimulationEvent[]
  /** Every session's events, for the all-runs view */
  sessionEvents?: ReadonlyMap<string, readonly SimulationEvent[]>
  sessionEventsVersion?: number
  /** Session list (for labels) and the one on screen, for the Compare view */
  sessions?: SessionInfo[]
  currentSessionId?: string | null
  selectedAgentId: string | null
  currentTime: number
  isPlaying: boolean
  onAgentClick: (agentId: string) => void
  onClose: () => void
}

/** Parent of an agent. Finished subagents leave the canvas, so fall back to their graph. */
function parentOf(id: string, graphs: Map<string, AgentGraph>, agents: Map<string, Agent>): string | null {
  return agents.get(id)?.parentId ?? graphs.get(id)?.parent ?? null
}

/** Graph to show: the selected agent's, else its nearest ancestor's, else the main agent's. */
export function pickGraphAgent(graphs: Map<string, AgentGraph>, agents: Map<string, Agent>, selectedAgentId: string | null): string | null {
  let id = selectedAgentId
  const seen = new Set<string>()
  while (id && !seen.has(id)) {
    if (graphs.has(id)) return id
    seen.add(id)
    id = parentOf(id, graphs, agents)
  }
  for (const agent of agents.values()) if (agent.isMain && graphs.has(agent.id)) return agent.id
  for (const graph of graphs.values()) if (graph.parent === null) return graph.agent
  return graphs.keys().next().value ?? null
}

/** Graph a subgraph node opens: its explicit child link, else the child agent named after it. */
function subgraphOf(graphs: Map<string, AgentGraph>, agents: Map<string, Agent>, parentId: string, node: GraphNodeInfo): string | null {
  if (node.child && graphs.has(node.child)) return node.child
  return childGraphFor(graphs, agents, parentId, node.label)
}

/**
 * Subagent graph behind a subgraph node (parallel runs are named "node #2", ...).
 * Prefers one still running, else the most recently started.
 */
function childGraphFor(graphs: Map<string, AgentGraph>, agents: Map<string, Agent>, parentId: string, nodeLabel: string): string | null {
  let best: string | null = null
  for (const id of graphs.keys()) {
    if (id !== nodeLabel && !id.startsWith(`${nodeLabel} #`)) continue
    if (parentOf(id, graphs, agents) !== parentId) continue
    const running = agents.get(id)?.state !== undefined && agents.get(id)?.state !== 'complete'
    const bestRunning = best !== null && agents.get(best)?.state !== undefined && agents.get(best)?.state !== 'complete'
    if (best === null || (running && !bestRunning) || running === bestRunning) best = id
  }
  return best
}

export const GraphPanel = memo(function GraphPanel({
  visible, graphs, agents, events, sessionEvents, sessionEventsVersion, sessions, currentSessionId, selectedAgentId, currentTime, isPlaying, onAgentClick, onClose,
}: GraphPanelProps) {
  const agentId = visible ? pickGraphAgent(graphs, agents, selectedAgentId) : null
  const runGraph = agentId ? graphs.get(agentId) ?? null : null
  const now = useSimulationClock(currentTime, isPlaying && visible)
  const [overlay, setOverlay] = useState<GraphOverlay>('runs')
  const [scope, setScope] = useState<Scope>('run')

  // Every run of this graph the UI has seen (sessions, and parallel instances)
  const aggregate = useMemo(
    () => (visible && runGraph && sessionEvents ? aggregateGraph(sessionEvents.values(), runGraph.agent, { costOf: agentCost }) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- recompute when any session's events grow, not on every render of this run's graph
    [visible, runGraph?.agent, sessionEvents, sessionEventsVersion],
  )
  const allRuns = scope === 'all' && aggregate !== null && aggregate.runs > 1

  // Compare: other sessions that ran this graph, most recent first
  const [baselineId, setBaselineId] = useState<string | null>(null)
  const candidates = useMemo(() => {
    if (!visible || !runGraph || !sessionEvents) return []
    const ids = sessionsWithGraph(sessionEvents, runGraph.agent, currentSessionId ?? null)
    const recency = new Map((sessions ?? []).map(s => [s.id, s.lastActivityTime]))
    return ids.sort((a, b) => (recency.get(b) ?? 0) - (recency.get(a) ?? 0))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- as for the aggregate
  }, [visible, runGraph?.agent, sessionEvents, sessionEventsVersion, currentSessionId, sessions])
  const baseline = baselineId && candidates.includes(baselineId) ? baselineId : candidates[0] ?? null
  const diff = useMemo(() => {
    if (scope !== 'compare' || !runGraph || !sessionEvents || !currentSessionId || !baseline) return null
    const current = sessionEvents.get(currentSessionId)
    const base = sessionEvents.get(baseline)
    return current && base ? diffRuns(current, base, runGraph.agent, { costOf: agentCost }) : null
    // eslint-disable-next-line react-hooks/exhaustive-deps -- as for the aggregate
  }, [scope, runGraph?.agent, sessionEvents, sessionEventsVersion, currentSessionId, baseline])
  const comparing = diff !== null

  const graph = comparing ? diff!.graph : allRuns ? aggregate!.graph : runGraph
  const layout = useMemo(() => (graph ? layoutGraph(graph) : null), [graph])
  const edgeShares = useMemo(
    () => (allRuns ? new Map([...aggregate!.edges].map(([id, e]) => [id, e.share])) : undefined),
    [allRuns, aggregate],
  )
  const cells = useMemo(() => {
    if (!visible || !graph) return new Map<string, OverlayCell>()
    if (comparing) return diffCells(diff!, overlay)
    if (allRuns) return aggregateCells(aggregate!, overlay)
    if (overlay === 'runs' || !events) return new Map<string, OverlayCell>()
    const metrics = analyzeRun(events, currentTime, { costOf: agentCost }).nodeMetrics.get(graph.agent)
    return overlayCells(metrics, overlay)
  }, [visible, graph, comparing, diff, allRuns, aggregate, overlay, events, currentTime])

  if (!visible) return null

  // Breadcrumb: ancestors of the viewed agent that have graphs of their own
  const trail: string[] = []
  for (let id: string | null = agentId; id && !trail.includes(id); id = parentOf(id, graphs, agents)) {
    if (graphs.has(id)) trail.unshift(id)
  }

  return (
    <SlidingPanel visible={visible} position={{ top: 48, right: 12 }} zIndex={Z.sidePanel} width={PANEL_W}>
      <div className="glass-card relative" {...stopPropagationHandlers}>
        <PanelHeader onClose={onClose}>
          <span className="text-[10px] font-mono tracking-wider" style={{ color: COLORS.textPrimary }}>
            GRAPH
          </span>
          {trail.length > 0 && (
            <span className="flex items-center gap-1 min-w-0 text-[9px] font-mono truncate">
              {trail.map((id, i) => (
                <span key={id} className="flex items-center gap-1 min-w-0">
                  {i > 0 && <span style={{ color: COLORS.textMuted }}>›</span>}
                  <button
                    onClick={() => onAgentClick(id)}
                    className="truncate hover:underline"
                    style={{ color: id === agentId ? COLORS.holoBright : COLORS.textDim, maxWidth: 120 }}
                    title={agents.get(id)?.name ?? id}
                  >
                    {id}
                  </button>
                </span>
              ))}
            </span>
          )}
        </PanelHeader>

        {!graph || !layout ? (
          <div className="text-[10px] font-mono py-4 text-center leading-relaxed" style={{ color: COLORS.textMuted }}>
            No graph data for this session.
            <br />
            Graph shape comes from framework adapters such as LangGraph.
          </div>
        ) : (
          <>
            {aggregate && aggregate.runs > 1 && (
              <div className="flex flex-wrap items-center gap-1 mb-1.5 text-[9px] font-mono" role="group" aria-label="Scope">
                {([['run', 'This run'], ['all', `All runs (${aggregate.runs})`], ...(candidates.length ? [['compare', 'Compare']] as const : [])] as const).map(([id, label]) => (
                  <button
                    key={id}
                    onClick={() => setScope(id)}
                    className="px-1.5 py-0.5 rounded"
                    title={id === 'all' ? `Every run of ${aggregate.name} seen in ${aggregate.sessions} session${aggregate.sessions === 1 ? '' : 's'}`
                      : id === 'compare' ? 'Diff this run against another run of this graph' : 'The selected session'}
                    style={{
                      color: scope === id ? COLORS.holoHot : COLORS.textMuted,
                      border: `1px solid ${scope === id ? COLORS.holoBase : COLORS.holoBorder06}`,
                      background: scope === id ? COLORS.holoBg10 : 'transparent',
                    }}
                    aria-pressed={scope === id}
                  >
                    {label}
                  </button>
                ))}
                {scope === 'compare' && candidates.length > 0 && (
                  <label className="flex items-center gap-1 min-w-0" style={{ color: COLORS.textMuted }}>
                    vs
                    <select
                      value={baseline ?? ''}
                      onChange={e => setBaselineId(e.target.value)}
                      className="font-mono text-[9px] rounded px-1 py-0.5 min-w-0"
                      style={{ background: COLORS.cardBgDark, color: COLORS.textPrimary, border: `1px solid ${COLORS.holoBorder06}`, maxWidth: 170 }}
                      title="Baseline run to compare with"
                    >
                      {candidates.map(id => (
                        <option key={id} value={id}>{sessions?.find(s => s.id === id)?.label ?? id}</option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            )}
            {events && (
              <div className="flex items-center gap-1 mb-2 text-[9px] font-mono" role="group" aria-label="Color nodes by">
                {OVERLAYS.map(o => (
                  <button
                    key={o.id}
                    onClick={() => setOverlay(o.id)}
                    title={o.title}
                    className="px-1.5 py-0.5 rounded"
                    style={{
                      color: overlay === o.id ? o.color : COLORS.textMuted,
                      border: `1px solid ${overlay === o.id ? o.color + '80' : COLORS.holoBorder06}`,
                      background: overlay === o.id ? o.color + '14' : 'transparent',
                    }}
                    aria-pressed={overlay === o.id}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            )}
            <div className="overflow-y-auto overflow-x-hidden" style={{ maxHeight: 'calc(100vh - 250px)' }}>
              <GraphSvg
                graph={graph}
                layout={layout}
                cells={cells}
                edgeShares={comparing ? undefined : edgeShares}
                edgeDiff={comparing ? diff!.edges : undefined}
                nodeDiff={comparing ? diff!.nodes : undefined}
                currentTime={now}
                onOpenSubgraph={(node) => {
                  const child = subgraphOf(graphs, agents, graph.agent, node)
                  if (child) onAgentClick(child)
                }}
                canOpen={(node) => subgraphOf(graphs, agents, graph.agent, node) !== null}
              />
            </div>
            {comparing
              ? <CompareFooter diff={diff!} baselineLabel={sessions?.find(s => s.id === baseline)?.label ?? baseline ?? ''} />
              : <GraphFooter graph={graph} layout={layout} overlay={overlay} cells={cells} aggregate={allRuns ? aggregate! : undefined} />}
          </>
        )}
      </div>
    </SlidingPanel>
  )
})

/**
 * The simulation time React sees only changes when events arrive, so after the
 * last event "just taken" highlights would never expire. While playing,
 * extrapolate from the last reported time with the wall clock; when paused
 * (review/scrub), hold the reported time.
 */
function useSimulationClock(currentTime: number, running: boolean): number {
  const anchor = useRef({ sim: currentTime, wall: 0 })
  const [, setTick] = useState(0)

  if (anchor.current.sim !== currentTime) anchor.current = { sim: currentTime, wall: performance.now() }

  useEffect(() => {
    if (!running) return
    if (!anchor.current.wall) anchor.current.wall = performance.now()
    const id = setInterval(() => setTick(t => t + 1), 250)
    return () => clearInterval(id)
  }, [running])

  if (!running || !anchor.current.wall) return currentTime
  return anchor.current.sim + (performance.now() - anchor.current.wall) / 1000
}

// ─── SVG ─────────────────────────────────────────────────────────────────────

interface GraphSvgProps {
  graph: AgentGraph
  layout: NonNullable<ReturnType<typeof layoutGraph>>
  cells: Map<string, OverlayCell>
  /** All-runs view: share of runs that took each edge */
  edgeShares?: Map<string, number>
  /** Compare view: whether each edge/node is in both runs, or only one */
  edgeDiff?: Map<string, DiffStatus>
  nodeDiff?: RunDiff['nodes']
  currentTime: number
  onOpenSubgraph: (node: GraphNodeInfo) => void
  canOpen: (node: GraphNodeInfo) => boolean
}

function GraphSvg({ graph, layout, cells, edgeShares, edgeDiff, nodeDiff, currentTime, onOpenSubgraph, canOpen }: GraphSvgProps) {
  const still = !!edgeShares || !!edgeDiff
  const edges = Object.values(graph.edges)
  // Draw untaken edges first so taken routes sit on top
  const ordered = [...edges].sort((a, b) => Math.min(1, a.traversals) - Math.min(1, b.traversals))

  return (
    // Scale wide graphs (e.g. parallel branches) down to the panel width; never up
    <svg
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      width="100%"
      style={{ display: 'block', margin: '0 auto', maxWidth: layout.width, height: 'auto' }}
      className="font-mono"
    >
      <style>{`
        @keyframes af-graph-flow { to { stroke-dashoffset: -20; } }
        @keyframes af-graph-pulse { 0%, 100% { opacity: 0.35; } 50% { opacity: 0.9; } }
        .af-graph-hot { stroke-dasharray: 6 4; animation: af-graph-flow 0.6s linear infinite; }
        .af-graph-running { animation: af-graph-pulse 1.2s ease-in-out infinite; }
      `}</style>
      <defs>
        {(['dim', 'lit', 'hot'] as const).map(tone => (
          <marker key={tone} id={`af-arrow-${tone}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={tone === 'dim' ? COLORS.textMuted : tone === 'lit' ? COLORS.holoBase : COLORS.holoBright} />
          </marker>
        ))}
      </defs>

      {ordered.map(edge => {
        const geo = layout.edges.get(edge.id)
        if (!geo) return null
        const hot = !still && edge.lastTime !== undefined && currentTime >= edge.lastTime && currentTime - edge.lastTime < HOT_HOP_S
        return <GraphEdgePath key={edge.id} geo={geo} traversals={edge.traversals} conditional={edge.conditional} hot={hot} label={edge.label} share={edgeShares ? edgeShares.get(edge.id) ?? 0 : undefined} diff={edgeDiff ? edgeDiff.get(edge.id) ?? null : undefined} />
      })}

      {graph.order.map(id => {
        const node = graph.nodes[id]
        const box = layout.nodes.get(id)
        if (!node || !box) return null
        const openable = node.kind === 'subgraph' && canOpen(node)
        return (
          <GraphNodeBox
            key={id}
            node={node}
            box={box}
            cell={cells.get(id)}
            aggregate={still}
            diff={nodeDiff?.get(id)?.status}
            openable={openable}
            onOpen={openable ? () => onOpenSubgraph(node) : undefined}
          />
        )
      })}
    </svg>
  )
}

function GraphEdgePath({ geo, traversals, conditional, hot, label, share, diff }: {
  geo: LayoutEdge; traversals: number; conditional: boolean; hot: boolean; label?: string
  /** All-runs view: share of runs that took this edge */
  share?: number
  /** Compare view: taken in both runs, only this one, only the baseline, or neither (null) */
  diff?: DiffStatus | null
}) {
  const taken = diff !== undefined ? diff !== null : share !== undefined ? share > 0 : traversals > 0
  const stroke = hot ? COLORS.holoBright
    : diff === 'added' ? COLORS.complete : diff === 'removed' ? COLORS.error
    : taken ? COLORS.holoBase : COLORS.textMuted
  const width = diff !== undefined ? (taken ? 2 : 1)
    : share !== undefined ? (taken ? 1 + 2.6 * share : 1) : taken ? Math.min(3.5, 1.4 + Math.log2(traversals)) : 1
  const takenText = diff !== undefined
    ? diff === 'added' ? 'taken in this run only' : diff === 'removed' ? 'taken in the baseline only' : 'taken in both runs'
    : share !== undefined ? `taken in ${Math.round(share * 100)}% of runs (${traversals}× in all)` : `taken ${traversals}×`
  const title = `${taken ? takenText : 'not taken'}${conditional ? ' · conditional' : ''}${geo.kind === 'back' || geo.kind === 'self' ? ' · loop' : ''}${label ? ` · ${label}` : ''}`
  const badge = diff !== undefined ? (diff === 'added' ? 'new' : diff === 'removed' ? 'gone' : null)
    : share !== undefined ? (taken && share < 1 ? `${Math.round(share * 100)}%` : null) : traversals > 1 ? `×${traversals}` : null
  const marker = diff === 'added' || diff === 'removed' ? 'lit' : hot ? 'hot' : taken ? 'lit' : 'dim'

  return (
    <g>
      <title>{title}</title>
      <path
        d={geo.path}
        fill="none"
        stroke={stroke}
        strokeWidth={width}
        strokeOpacity={taken ? 0.9 : 0.8}
        strokeDasharray={diff === 'removed' ? '4 3' : !hot && conditional && !taken ? '3 3' : undefined}
        className={hot ? 'af-graph-hot' : undefined}
        markerEnd={`url(#af-arrow-${marker})`}
      />
      {badge && (
        <g transform={`translate(${geo.labelX}, ${geo.labelY})`}>
          <rect x={-13} y={-7} width={26} height={14} rx={7} fill={COLORS.void} stroke={stroke} strokeOpacity={0.6} />
          <text textAnchor="middle" dominantBaseline="central" fontSize={8} fill={diff === 'added' ? COLORS.complete : diff === 'removed' ? COLORS.error : COLORS.holoBright}>{badge}</text>
        </g>
      )}
    </g>
  )
}

function GraphNodeBox({ node, box, cell, aggregate, diff, openable, onOpen }: {
  node: GraphNodeInfo; box: LayoutNode; cell?: OverlayCell; aggregate?: boolean; diff?: DiffStatus; openable: boolean; onOpen?: () => void
}) {
  const visited = node.visits > 0
  const running = !aggregate && node.running > 0
  const terminal = node.kind === 'start' || node.kind === 'end'
  const stroke = diff === 'added' ? COLORS.complete
    : diff === 'removed' ? COLORS.error
    : node.error && !aggregate ? COLORS.error
    : running ? COLORS.tool_calling
    : visited ? COLORS.complete
    : COLORS.textMuted
  const rx = terminal ? box.h / 2 : 5
  const label = terminal ? (node.kind === 'start' ? 'START' : 'END') : node.label
  const title = [
    node.label,
    node.kind === 'subgraph' ? (openable ? 'subgraph — click to open' : 'subgraph (not run yet)') : null,
    aggregate ? (visited ? null : 'never ran') : visited ? `ran ${node.visits}×${node.lastStep !== undefined ? ` · last step ${node.lastStep}` : ''}` : 'not run',
    running ? 'running now' : null,
    node.declared ? null : 'observed at runtime (not in declared graph)',
    node.error ? `error: ${node.error}` : null,
    cell?.detail ?? (cell ? `${cell.text}${cell.heat < 1 ? ` · ${Math.round(cell.heat * 100)}% of the top node` : ' · the top node'}` : null),
  ].filter(Boolean).join('\n')
  const pillW = cell ? cell.text.length * 5 + 8 : 0
  // Wide pills sit centered over the node so they don't cover its neighbors
  const pillX = pillW > box.w * 0.6 ? (box.w - pillW) / 2 : box.w - pillW + 4

  return (
    <g
      transform={`translate(${box.x}, ${box.y})`}
      onClick={onOpen}
      style={{ cursor: onOpen ? 'pointer' : 'default' }}
      opacity={diff === 'removed' ? 0.7 : 1}
    >
      <title>{title}</title>
      {node.kind === 'subgraph' && (
        <rect x={3} y={3} width={box.w} height={box.h} rx={rx} fill={COLORS.nodeInterior} stroke={stroke} strokeOpacity={0.4} />
      )}
      {running && (
        <rect x={-3} y={-3} width={box.w + 6} height={box.h + 6} rx={rx + 3} fill="none" stroke={COLORS.tool_calling} strokeWidth={2} className="af-graph-running" />
      )}
      <rect
        width={box.w}
        height={box.h}
        rx={rx}
        fill={terminal ? COLORS.holoBg10 : COLORS.cardBgDark}
        stroke={stroke}
        strokeWidth={visited || running ? 1.4 : 1}
        strokeDasharray={diff === 'removed' ? '4 2' : !visited && !node.declared ? '3 2' : undefined}
      />
      {cell && (
        <rect width={box.w} height={box.h} rx={rx} fill={cell.color} fillOpacity={0.06 + 0.5 * cell.heat} stroke={cell.color} strokeOpacity={0.25 + 0.6 * cell.heat} strokeWidth={cell.heat > 0.66 ? 1.6 : 1} />
      )}
      <text
        x={box.w / 2}
        y={box.h / 2}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={terminal ? 8 : 10}
        letterSpacing={terminal ? 1 : 0}
        fill={visited ? COLORS.textPrimary : COLORS.textDim}
      >
        {truncateLabel(label, box.w)}
        {node.kind === 'subgraph' ? ' ▸' : ''}
      </text>
      {/* Visit count on the left corner: loops leave from the right side */}
      {node.visits > 1 && !terminal && !aggregate && (
        <g transform={`translate(2, 0)`}>
          <circle r={8} fill={COLORS.void} stroke={stroke} strokeWidth={1} />
          <text textAnchor="middle" dominantBaseline="central" fontSize={7.5} fill={COLORS.textPrimary}>×{node.visits}</text>
        </g>
      )}
      {cell && !terminal && (
        <g transform={`translate(${pillX}, ${-6})`}>
          <rect width={pillW} height={12} rx={6} fill={COLORS.void} stroke={cell.color} strokeOpacity={0.8} />
          <text x={pillW / 2} y={6} textAnchor="middle" dominantBaseline="central" fontSize={7.5} fill={cell.color}>{cell.text}</text>
        </g>
      )}
    </g>
  )
}

function truncateLabel(label: string, width: number): string {
  const max = Math.max(4, Math.floor((width - 18) / 6.2))
  return label.length > max ? label.slice(0, max - 1) + '…' : label
}

// ─── Footer ──────────────────────────────────────────────────────────────────

function GraphFooter({ graph, layout, overlay, cells, aggregate }: {
  graph: AgentGraph; layout: NonNullable<ReturnType<typeof layoutGraph>>; overlay: GraphOverlay; cells: Map<string, OverlayCell>
  aggregate?: GraphAggregate
}) {
  let top: [string, OverlayCell] | null = null
  let total = 0
  for (const entry of cells) {
    total += entry[1].value
    if (!top || entry[1].value > top[1].value) top = entry
  }
  const real = Object.values(graph.nodes).filter(n => n.kind !== 'start' && n.kind !== 'end')
  const ran = real.filter(n => n.visits > 0).length
  let loops = 0
  for (const geo of layout.edges.values()) {
    if (geo.kind !== 'forward') loops += graph.edges[geo.id]?.traversals ?? 0
  }

  return (
    <div className="mt-2 pt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] font-mono" style={{ borderTop: `1px solid ${COLORS.holoBorder06}`, color: COLORS.textMuted }}>
      <span style={{ color: COLORS.textDim }}>
        {aggregate
          ? `${aggregate.runs} runs · ${aggregate.sessions} session${aggregate.sessions === 1 ? '' : 's'} · ${ran}/${real.length} nodes ever ran`
          : `${ran}/${real.length} nodes · ${graph.totalHops} hops · step ${graph.lastStep}${loops > 0 ? ` · ${loops} loop${loops === 1 ? '' : 's'}` : ''}`}
      </span>
      <span className="flex-1" />
      <LegendLine color={COLORS.holoBase} width={2} label="taken" />
      <LegendLine color={COLORS.textMuted} width={1} label="not taken" />
      <LegendLine color={COLORS.textMuted} width={1} dashed label="conditional" />
      {!aggregate && (
        <span className="flex items-center gap-1">
          <svg width={10} height={10}><rect x={1} y={1} width={8} height={8} rx={2} fill="none" stroke={COLORS.tool_calling} strokeWidth={1.5} /></svg>
          running
        </span>
      )}
      {aggregate && (
        <span className="w-full" style={{ color: COLORS.textDim }}>
          {top
            ? overlay === 'runs' ? <>Edges and nodes show the share of runs that took them.</>
              : <>{overlay === 'time' ? 'Slowest (median/p95)' : overlay === 'errors' ? 'Fails most often' : overlay === 'cost' ? 'Costliest per run' : 'Most tokens (median)'}: <span style={{ color: top[1].color }}>{top[0]}</span> {top[1].text}</>
            : `No ${overlay} recorded for these nodes.`}
        </span>
      )}
      {!aggregate && overlay !== 'runs' && (
        <span className="w-full" style={{ color: COLORS.textDim }}>
          {top && total > 0
            ? <>Most {overlay === 'errors' ? 'errors' : overlay}: <span style={{ color: top[1].color }}>{top[0]}</span> {top[1].text}{cells.size > 1 ? ` (${Math.round((top[1].value / total) * 100)}% of all nodes)` : ''}</>
            : `No ${overlay} recorded for these nodes yet.`}
        </span>
      )}
      {!graph.hasStructure && (
        <span className="w-full" style={{ color: COLORS.textFaint }}>
          Showing observed routes only. Pass <code>graph=</code> to the adapter to see every declared route.
        </span>
      )}
    </div>
  )
}

const COMPARE_ROWS: { key: keyof RunTotals; label: string; format: (v: number) => string }[] = [
  { key: 'duration', label: 'Duration', format: v => formatMetric(v, 'time') },
  { key: 'cost', label: 'Est. cost', format: v => formatMetric(v, 'cost') },
  { key: 'tokens', label: 'Tokens', format: v => formatMetric(v, 'tokens') },
  { key: 'toolCalls', label: 'Tool calls', format: v => String(v) },
  { key: 'errors', label: 'Errors', format: v => String(v) },
  { key: 'agents', label: 'Agents', format: v => String(v) },
]

/** Compare view: the two runs overall, and how the routes changed. */
function CompareFooter({ diff, baselineLabel }: { diff: RunDiff; baselineLabel: string }) {
  const added = [...diff.edges.values()].filter(s => s === 'added').length
  const removed = [...diff.edges.values()].filter(s => s === 'removed').length
  return (
    <div className="mt-2 pt-2 text-[9px] font-mono" style={{ borderTop: `1px solid ${COLORS.holoBorder06}`, color: COLORS.textMuted }}>
      <div className="grid gap-x-3 gap-y-0.5" style={{ gridTemplateColumns: 'auto 1fr 1fr auto' }}>
        <span />
        <span style={{ color: COLORS.textDim }}>this run</span>
        <span className="truncate" style={{ color: COLORS.textDim }} title={baselineLabel}>baseline</span>
        <span style={{ color: COLORS.textDim, textAlign: 'right' }}>change</span>
        {COMPARE_ROWS.map(row => {
          const cur = diff.current[row.key]
          const base = diff.baseline[row.key]
          const delta = cur - base
          const worse = delta > (row.key === 'duration' ? 0.01 : 0)
          const better = delta < (row.key === 'duration' ? -0.01 : 0)
          // More of anything here is worse, except agents, which is neutral
          const color = row.key === 'agents' ? COLORS.textDim : worse ? COLORS.error : better ? COLORS.complete : COLORS.textMuted
          const pct = base > 0 ? ` (${delta > 0 ? '+' : ''}${Math.round((delta / base) * 100)}%)` : ''
          return [
            <span key={`${row.key}-l`}>{row.label}</span>,
            <span key={`${row.key}-c`} style={{ color: COLORS.textPrimary }}>{row.format(cur)}</span>,
            <span key={`${row.key}-b`}>{row.format(base)}</span>,
            <span key={`${row.key}-d`} style={{ color, textAlign: 'right' }}>
              {Math.abs(delta) < (row.key === 'duration' ? 0.01 : 1e-9) ? '—' : `${delta > 0 ? '+' : '\u2212'}${row.format(Math.abs(delta))}${pct}`}
            </span>,
          ]
        })}
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span>
          Routes: {added || removed
            ? <><span style={{ color: COLORS.complete }}>{added} new</span> · <span style={{ color: COLORS.error }}>{removed} gone</span></>
            : 'same as the baseline'}
        </span>
        <span className="flex-1" />
        <LegendLine color={COLORS.complete} width={2} label="this run only" />
        <LegendLine color={COLORS.error} width={2} dashed label="baseline only" />
      </div>
    </div>
  )
}

function LegendLine({ color, width, dashed, label }: { color: string; width: number; dashed?: boolean; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <svg width={16} height={6}><line x1={0} y1={3} x2={16} y2={3} stroke={color} strokeWidth={width} strokeDasharray={dashed ? '3 2' : undefined} /></svg>
      {label}
    </span>
  )
}
