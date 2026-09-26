'use client'

import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { Z, type Agent, type AgentGraph, type GraphNodeInfo } from '@/lib/agent-types'
import { COLORS } from '@/lib/colors'
import { layoutGraph, type LayoutEdge, type LayoutNode } from '@/lib/graph-layout'
import { PanelHeader, SlidingPanel, stopPropagationHandlers } from './shared-ui'

/** How long (seconds) a just-taken hop keeps its flowing highlight */
const HOT_HOP_S = 1.5
// Shrinks on narrow windows (e.g. a VS Code side column)
const PANEL_W = 'min(380px, calc(100vw - 24px))'

interface GraphPanelProps {
  visible: boolean
  graphs: Map<string, AgentGraph>
  agents: Map<string, Agent>
  selectedAgentId: string | null
  currentTime: number
  isPlaying: boolean
  onAgentClick: (agentId: string) => void
  onClose: () => void
}

/** Graph to show: the selected agent's, else its nearest ancestor's, else the main agent's. */
export function pickGraphAgent(graphs: Map<string, AgentGraph>, agents: Map<string, Agent>, selectedAgentId: string | null): string | null {
  let id = selectedAgentId
  const seen = new Set<string>()
  while (id && !seen.has(id)) {
    if (graphs.has(id)) return id
    seen.add(id)
    id = agents.get(id)?.parentId ?? null
  }
  for (const agent of agents.values()) if (agent.isMain && graphs.has(agent.id)) return agent.id
  return graphs.keys().next().value ?? null
}

/** Child agent that ran a subgraph node (parallel runs are named "node #2", ...). */
function childAgentFor(agents: Map<string, Agent>, parentId: string, nodeLabel: string): Agent | null {
  let best: Agent | null = null
  for (const agent of agents.values()) {
    if (agent.parentId !== parentId) continue
    if (agent.id !== nodeLabel && !agent.id.startsWith(`${nodeLabel} #`)) continue
    if (!best || (agent.state !== 'complete' && best.state === 'complete') || agent.spawnTime > best.spawnTime) best = agent
  }
  return best
}

export const GraphPanel = memo(function GraphPanel({
  visible, graphs, agents, selectedAgentId, currentTime, isPlaying, onAgentClick, onClose,
}: GraphPanelProps) {
  const agentId = visible ? pickGraphAgent(graphs, agents, selectedAgentId) : null
  const graph = agentId ? graphs.get(agentId) ?? null : null
  const layout = useMemo(() => (graph ? layoutGraph(graph) : null), [graph])
  const now = useSimulationClock(currentTime, isPlaying && visible)

  if (!visible) return null

  // Breadcrumb: ancestors of the viewed agent that have graphs of their own
  const trail: Agent[] = []
  for (let id: string | null = agentId; id; id = agents.get(id)?.parentId ?? null) {
    const agent = agents.get(id)
    if (!agent || trail.includes(agent)) break
    if (graphs.has(id)) trail.unshift(agent)
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
              {trail.map((agent, i) => (
                <span key={agent.id} className="flex items-center gap-1 min-w-0">
                  {i > 0 && <span style={{ color: COLORS.textMuted }}>›</span>}
                  <button
                    onClick={() => onAgentClick(agent.id)}
                    className="truncate hover:underline"
                    style={{ color: agent.id === agentId ? COLORS.holoBright : COLORS.textDim, maxWidth: 120 }}
                    title={agent.name}
                  >
                    {agent.isMain ? agent.id : agent.name}
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
            <div className="overflow-auto" style={{ maxHeight: 'calc(100vh - 220px)' }}>
              <GraphSvg
                graph={graph}
                layout={layout}
                currentTime={now}
                onOpenSubgraph={(label) => {
                  const child = childAgentFor(agents, graph.agent, label)
                  if (child) onAgentClick(child.id)
                }}
                canOpen={(label) => childAgentFor(agents, graph.agent, label) !== null}
              />
            </div>
            <GraphFooter graph={graph} layout={layout} />
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
  currentTime: number
  onOpenSubgraph: (label: string) => void
  canOpen: (label: string) => boolean
}

function GraphSvg({ graph, layout, currentTime, onOpenSubgraph, canOpen }: GraphSvgProps) {
  const edges = Object.values(graph.edges)
  // Draw untaken edges first so taken routes sit on top
  const ordered = [...edges].sort((a, b) => Math.min(1, a.traversals) - Math.min(1, b.traversals))

  return (
    <svg width={layout.width} height={layout.height} style={{ display: 'block', margin: '0 auto' }} className="font-mono">
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
        const hot = edge.lastTime !== undefined && currentTime >= edge.lastTime && currentTime - edge.lastTime < HOT_HOP_S
        return <GraphEdgePath key={edge.id} geo={geo} traversals={edge.traversals} conditional={edge.conditional} hot={hot} label={edge.label} />
      })}

      {graph.order.map(id => {
        const node = graph.nodes[id]
        const box = layout.nodes.get(id)
        if (!node || !box) return null
        const openable = node.kind === 'subgraph' && canOpen(node.label)
        return (
          <GraphNodeBox
            key={id}
            node={node}
            box={box}
            openable={openable}
            onOpen={openable ? () => onOpenSubgraph(node.label) : undefined}
          />
        )
      })}
    </svg>
  )
}

function GraphEdgePath({ geo, traversals, conditional, hot, label }: {
  geo: LayoutEdge; traversals: number; conditional: boolean; hot: boolean; label?: string
}) {
  const taken = traversals > 0
  const stroke = hot ? COLORS.holoBright : taken ? COLORS.holoBase : COLORS.textMuted
  const width = taken ? Math.min(3.5, 1.4 + Math.log2(traversals)) : 1
  const title = `${taken ? `taken ${traversals}×` : 'not taken'}${conditional ? ' · conditional' : ''}${geo.kind === 'back' || geo.kind === 'self' ? ' · loop' : ''}${label ? ` · ${label}` : ''}`

  return (
    <g>
      <title>{title}</title>
      <path
        d={geo.path}
        fill="none"
        stroke={stroke}
        strokeWidth={width}
        strokeOpacity={taken ? 0.9 : 0.8}
        strokeDasharray={!hot && conditional && !taken ? '3 3' : undefined}
        className={hot ? 'af-graph-hot' : undefined}
        markerEnd={`url(#af-arrow-${hot ? 'hot' : taken ? 'lit' : 'dim'})`}
      />
      {traversals > 1 && (
        <g transform={`translate(${geo.labelX}, ${geo.labelY})`}>
          <rect x={-11} y={-7} width={22} height={14} rx={7} fill={COLORS.void} stroke={stroke} strokeOpacity={0.6} />
          <text textAnchor="middle" dominantBaseline="central" fontSize={8} fill={COLORS.holoBright}>×{traversals}</text>
        </g>
      )}
    </g>
  )
}

function GraphNodeBox({ node, box, openable, onOpen }: {
  node: GraphNodeInfo; box: LayoutNode; openable: boolean; onOpen?: () => void
}) {
  const visited = node.visits > 0
  const running = node.running > 0
  const terminal = node.kind === 'start' || node.kind === 'end'
  const stroke = node.error ? COLORS.error
    : running ? COLORS.tool_calling
    : visited ? COLORS.complete
    : COLORS.textMuted
  const rx = terminal ? box.h / 2 : 5
  const label = terminal ? (node.kind === 'start' ? 'START' : 'END') : node.label
  const title = [
    node.label,
    node.kind === 'subgraph' ? (openable ? 'subgraph — click to open' : 'subgraph (not run yet)') : null,
    visited ? `ran ${node.visits}×${node.lastStep !== undefined ? ` · last step ${node.lastStep}` : ''}` : 'not run',
    running ? 'running now' : null,
    node.declared ? null : 'observed at runtime (not in declared graph)',
    node.error ? `error: ${node.error}` : null,
  ].filter(Boolean).join('\n')

  return (
    <g
      transform={`translate(${box.x}, ${box.y})`}
      onClick={onOpen}
      style={{ cursor: onOpen ? 'pointer' : 'default' }}
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
        strokeDasharray={!visited && !node.declared ? '3 2' : undefined}
      />
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
      {node.visits > 1 && !terminal && (
        <g transform={`translate(2, 0)`}>
          <circle r={8} fill={COLORS.void} stroke={stroke} strokeWidth={1} />
          <text textAnchor="middle" dominantBaseline="central" fontSize={7.5} fill={COLORS.textPrimary}>×{node.visits}</text>
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

function GraphFooter({ graph, layout }: { graph: AgentGraph; layout: NonNullable<ReturnType<typeof layoutGraph>> }) {
  const real = Object.values(graph.nodes).filter(n => n.kind !== 'start' && n.kind !== 'end')
  const ran = real.filter(n => n.visits > 0).length
  let loops = 0
  for (const geo of layout.edges.values()) {
    if (geo.kind !== 'forward') loops += graph.edges[geo.id]?.traversals ?? 0
  }

  return (
    <div className="mt-2 pt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] font-mono" style={{ borderTop: `1px solid ${COLORS.holoBorder06}`, color: COLORS.textMuted }}>
      <span style={{ color: COLORS.textDim }}>
        {ran}/{real.length} nodes · {graph.totalHops} hops · step {graph.lastStep}{loops > 0 ? ` · ${loops} loop${loops === 1 ? '' : 's'}` : ''}
      </span>
      <span className="flex-1" />
      <LegendLine color={COLORS.holoBase} width={2} label="taken" />
      <LegendLine color={COLORS.textMuted} width={1} label="not taken" />
      <LegendLine color={COLORS.textMuted} width={1} dashed label="conditional" />
      <span className="flex items-center gap-1">
        <svg width={10} height={10}><rect x={1} y={1} width={8} height={8} rx={2} fill="none" stroke={COLORS.tool_calling} strokeWidth={1.5} /></svg>
        running
      </span>
      {!graph.hasStructure && (
        <span className="w-full" style={{ color: COLORS.textFaint }}>
          Showing observed routes only. Pass <code>graph=</code> to the adapter to see every declared route.
        </span>
      )}
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
