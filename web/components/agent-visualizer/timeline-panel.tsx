'use client'

import { useRef, useEffect, useMemo, useState, useCallback } from 'react'
import { Z, type Agent, type SimulationEvent } from '@/lib/agent-types'
import { COLORS, contextSegments } from '@/lib/colors'
import { modelColor } from '@/lib/model-colors'
import { analyzeRun, type RunAnalysis } from '@/lib/run-analysis'
import { formatTokens } from '@/lib/utils'
import type { ConversationMessage } from '@/hooks/simulation/types'
import { PanelHeader, SlidingPanel } from './shared-ui'

interface TimelinePanelProps {
  visible: boolean
  events: readonly SimulationEvent[]
  currentTime: number
  /** End of the run so far: the timeline keeps its full width while you scrub back */
  maxTime?: number
  /** Agents and conversations as they stood at currentTime, for the moment strip */
  agents?: Map<string, Agent>
  conversations?: Map<string, ConversationMessage[]>
  /** Seek to a time (dragging the ruler or the playhead) */
  onSeek?: (time: number) => void
  onAgentClick?: (agentId: string) => void
  onClose: () => void
}

// ─── Layout constants ────────────────────────────────────────────────────────

const HEADER_HEIGHT = 20
const LABEL_WIDTH = 128
const SUB_ROW = 12
const LANE_PAD = 4
const INDENT = 8
const FONT = '9px monospace'

const laneHeight = (rows: number) => rows * SUB_ROW + LANE_PAD * 2

/** x ↔ time mapping shared by drawing and dragging */
function timeScale(a: RunAnalysis, width: number) {
  const barWidth = width - LABEL_WIDTH - 8
  const span = Math.max(a.end - a.start, 0.001)
  return {
    x: (t: number) => LABEL_WIDTH + ((t - a.start) / span) * barWidth,
    t: (x: number) => a.start + Math.min(1, Math.max(0, (x - LABEL_WIDTH) / barWidth)) * span,
  }
}

const LEGEND_ITEMS = [
  { color: COLORS.holoBase, label: 'Agent running' },
  { color: COLORS.tool, label: 'Tool call' },
  { color: COLORS.error, label: 'Error' },
  { color: COLORS.holoHot, label: 'Critical path' },
]

function formatSeconds(s: number): string {
  if (s < 1) return `${Math.round(s * 1000)}ms`
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`
  return `${Math.floor(s / 60)}m${Math.round(s % 60).toString().padStart(2, '0')}s`
}

interface HitRect { x: number; y: number; w: number; h: number; text: string; lane: string }

// ─── Canvas-based swimlane rendering ─────────────────────────────────────────

function drawSwimlanes(
  ctx: CanvasRenderingContext2D,
  a: RunAnalysis,
  currentTime: number,
  showCritical: boolean,
  width: number,
  height: number,
  dpr: number,
): HitRect[] {
  const hits: HitRect[] = []
  ctx.clearRect(0, 0, width * dpr, height * dpr)
  ctx.save()
  ctx.scale(dpr, dpr)
  ctx.font = FONT

  if (a.lanes.length === 0) {
    ctx.fillStyle = COLORS.textMuted
    ctx.textAlign = 'center'
    ctx.fillText('No timeline data', width / 2, height / 2)
    ctx.restore()
    return hits
  }

  const t0 = a.start
  const span = Math.max(a.end - t0, 0.001)
  const barWidth = width - LABEL_WIDTH - 8
  const xOf = (t: number) => LABEL_WIDTH + ((t - t0) / span) * barWidth

  // Time markers
  const rough = span / 8
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
  const interval = steps.find(s => s >= rough) ?? 600
  const markers: number[] = []
  for (let t = Math.ceil(t0 / interval) * interval; t <= a.end + 1e-9; t += interval) markers.push(t)

  ctx.textAlign = 'center'
  ctx.fillStyle = COLORS.textMuted
  for (const t of markers) ctx.fillText(formatSeconds(t - t0), xOf(t), HEADER_HEIGHT - 6)

  // Lane positions
  const laneY = new Map<string, { y: number; h: number }>()
  let y = HEADER_HEIGHT
  for (const lane of a.lanes) {
    const h = laneHeight(lane.rows)
    laneY.set(lane.name, { y, h })
    y += h
  }

  const dim = (critical: boolean) => (showCritical && !critical ? 0.28 : 1)

  for (const lane of a.lanes) {
    const { y: ly, h } = laneY.get(lane.name)!
    const onPath = a.criticalLanes.has(lane.name)

    // Label, indented by depth
    ctx.textAlign = 'left'
    ctx.globalAlpha = dim(onPath)
    ctx.fillStyle = lane.errors ? COLORS.error : onPath && showCritical ? COLORS.textPrimary : COLORS.textDim
    const indent = Math.min(lane.depth, 6) * INDENT
    const maxChars = Math.floor((LABEL_WIDTH - 8 - indent) / 5.4)
    const name = lane.name.length > maxChars ? lane.name.slice(0, Math.max(1, maxChars - 1)) + '…' : lane.name
    ctx.fillText(`${lane.depth ? '└ ' : ''}${name}`, 4 + indent, ly + Math.min(h, 20) / 2 + 3)

    // Track and marker lines
    ctx.globalAlpha = 1
    ctx.fillStyle = COLORS.holoBg03
    ctx.fillRect(LABEL_WIDTH, ly + 2, barWidth, h - 4)
    ctx.fillStyle = COLORS.panelSeparator
    for (const t of markers) ctx.fillRect(xOf(t), ly + 2, 1, h - 4)

    // When the agent was running
    for (const interval of lane.intervals) {
      const x1 = xOf(interval.start)
      const x2 = xOf(interval.end ?? a.end)
      ctx.globalAlpha = 0.14 * dim(onPath)
      ctx.fillStyle = COLORS.holoBase
      ctx.fillRect(x1, ly + 2, Math.max(1, x2 - x1), h - 4)
      ctx.globalAlpha = 0.5 * dim(onPath)
      ctx.fillRect(x1, ly + 2, 1, h - 4)
      hits.push({
        x: x1, y: ly + 2, w: Math.max(2, x2 - x1), h: h - 4, lane: lane.name,
        text: `${lane.name}${lane.model ? ` · ${lane.model}` : ''}\nran ${formatSeconds((interval.end ?? a.end) - interval.start)}${interval.end === undefined ? ' (running)' : ''}${lane.tokens ? ` · ${lane.tokens.toLocaleString()} tokens` : ''}`,
      })
    }

    // Tool calls, parallel ones stacked
    for (const tool of lane.tools) {
      const x1 = xOf(tool.start)
      const w = Math.max(2, xOf(tool.end ?? a.end) - x1)
      const ty = ly + LANE_PAD + tool.row * SUB_ROW + 1
      const th = SUB_ROW - 2
      const critical = a.criticalToolIds.has(tool.id)
      const color = tool.error ? COLORS.error : COLORS.tool
      ctx.globalAlpha = (critical && showCritical ? 0.6 : 0.35) * dim(critical)
      ctx.fillStyle = color
      ctx.fillRect(x1, ty, w, th)
      ctx.globalAlpha = dim(critical)
      ctx.strokeStyle = critical && showCritical ? COLORS.holoHot : color
      ctx.lineWidth = critical && showCritical ? 1.2 : 0.6
      ctx.strokeRect(x1 + 0.5, ty + 0.5, w - 1, th - 1)
      if (w > 36) {
        ctx.save()
        ctx.beginPath()
        ctx.rect(x1, ty, w, th)
        ctx.clip()
        ctx.fillStyle = critical && showCritical ? COLORS.holoHot : color
        ctx.globalAlpha = 0.9 * dim(critical)
        ctx.fillText(tool.label, x1 + 3, ty + th / 2 + 3)
        ctx.restore()
      }
      hits.push({
        x: x1, y: ty, w, h: th, lane: lane.name,
        text: `${tool.label}\n${lane.name} · ${formatSeconds((tool.end ?? a.end) - tool.start)}${tool.end === undefined ? ' (running)' : ''}${tool.error ? `\nerror: ${tool.error}` : ''}${critical ? '\non the critical path' : ''}`,
      })
    }
    ctx.globalAlpha = 1
  }

  // Critical path: the agent's own time as a bright rail, hops between lanes as links
  if (showCritical) {
    ctx.strokeStyle = COLORS.holoHot
    ctx.fillStyle = COLORS.holoHot
    let prev: { x: number; y: number } | null = null
    for (const seg of a.critical) {
      const pos = laneY.get(seg.lane)
      if (!pos) continue
      const railY = pos.y + pos.h - 3
      const x1 = xOf(seg.start)
      const x2 = xOf(seg.end)
      if (prev && Math.abs(prev.y - railY) > 1) {
        ctx.globalAlpha = 0.5
        ctx.setLineDash([2, 2])
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(prev.x, prev.y)
        ctx.lineTo(x1, railY)
        ctx.stroke()
        ctx.setLineDash([])
      }
      if (seg.kind === 'self') {
        ctx.globalAlpha = 0.85
        ctx.fillRect(x1, railY - 1, Math.max(1, x2 - x1), 2)
        hits.push({ x: x1, y: railY - 3, w: Math.max(2, x2 - x1), h: 6, lane: seg.lane, text: `${seg.lane} itself (model calls, thinking)\n${formatSeconds(seg.end - seg.start)} on the critical path` })
      }
      prev = { x: x2, y: railY }
    }
    ctx.globalAlpha = 1
  }

  // What hasn't happened yet at the playhead is dimmed
  const px = xOf(Math.min(Math.max(currentTime, t0), a.end))
  if (px < LABEL_WIDTH + barWidth - 1) {
    ctx.fillStyle = COLORS.void
    ctx.globalAlpha = 0.55
    ctx.fillRect(px, HEADER_HEIGHT, LABEL_WIDTH + barWidth - px, height - HEADER_HEIGHT)
  }

  // Playhead, with a handle on the ruler to drag
  ctx.fillStyle = COLORS.holoHot
  ctx.globalAlpha = 0.7
  ctx.fillRect(px, HEADER_HEIGHT - 4, 1, height - HEADER_HEIGHT + 4)
  ctx.globalAlpha = 1
  ctx.beginPath()
  ctx.moveTo(px - 4, HEADER_HEIGHT - 9)
  ctx.lineTo(px + 4, HEADER_HEIGHT - 9)
  ctx.lineTo(px, HEADER_HEIGHT - 3)
  ctx.closePath()
  ctx.fill()

  ctx.restore()
  return hits
}

// ─── Component ──────────────────────────────────────────────────────────────

export function TimelinePanel({ visible, events, currentTime, maxTime, agents, conversations, onSeek, onAgentClick, onClose }: TimelinePanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const hitsRef = useRef<HitRect[]>([])
  const [showCritical, setShowCritical] = useState(true)
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null)
  const [dragging, setDragging] = useState(false)

  // Lay out the whole run so far, so scrubbing back keeps the full width. It
  // ends at the last event (the live clock keeps ticking after a run ends),
  // or at the clock while agents are still running.
  const lastEvent = events.length ? events[events.length - 1].time : 0
  const live = maxTime === undefined || currentTime >= maxTime - 1e-6
  const running = agents ? [...agents.values()].some(a => a.state !== 'complete') : true
  const horizon = live && running ? Math.max(currentTime, lastEvent) : lastEvent
  const analysis = useMemo(() => (visible ? analyzeRun(events, horizon) : null), [visible, events, horizon])

  // Dragging the ruler or the playhead seeks, at most once per frame
  const pendingSeek = useRef<number | null>(null)
  const seekFrame = useRef<number | null>(null)
  const seekTo = useCallback((clientX: number) => {
    const canvas = canvasRef.current
    if (!canvas || !analysis || !onSeek) return
    const x = clientX - canvas.getBoundingClientRect().left
    pendingSeek.current = timeScale(analysis, canvas.clientWidth).t(x)
    if (seekFrame.current !== null) return
    seekFrame.current = requestAnimationFrame(() => {
      seekFrame.current = null
      if (pendingSeek.current !== null) onSeek(pendingSeek.current)
    })
  }, [analysis, onSeek])
  useEffect(() => {
    if (!dragging) return
    const move = (e: MouseEvent) => seekTo(e.clientX)
    const up = () => setDragging(false)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [dragging, seekTo])
  useEffect(() => () => { if (seekFrame.current !== null) cancelAnimationFrame(seekFrame.current) }, [])

  /** On the ruler, or on the playhead line: a place to grab for scrubbing */
  const onScrubHandle = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!analysis || !onSeek) return false
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    if (x < LABEL_WIDTH) return false
    const px = timeScale(analysis, e.currentTarget.clientWidth).x(Math.min(currentTime, analysis.end))
    return y <= HEADER_HEIGHT || Math.abs(x - px) <= 4
  }, [analysis, onSeek, currentTime])
  const canvasHeight = HEADER_HEIGHT + (analysis?.lanes.reduce((sum, l) => sum + laneHeight(l.rows), 0) ?? 0) + 4

  useEffect(() => {
    if (!visible || !analysis) return
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    // Use the scroll container's clientWidth (excludes scrollbar) for a snug fit
    const width = canvas.parentElement?.clientWidth ?? canvas.clientWidth
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = canvasHeight * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${canvasHeight}px`
    hitsRef.current = drawSwimlanes(ctx, analysis, currentTime, showCritical, width, canvasHeight, dpr)
  }, [visible, analysis, currentTime, canvasHeight, showCritical])

  const hitAt = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    // Last drawn wins: tool calls sit on top of lanes
    for (let i = hitsRef.current.length - 1; i >= 0; i--) {
      const h = hitsRef.current[i]
      if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) return { hit: h, x, y }
    }
    return null
  }, [])

  if (!visible || !analysis) return null

  const total = analysis.breakdown.tools + analysis.breakdown.self
  const toolShare = total > 0 ? Math.round((analysis.breakdown.tools / total) * 100) : 0

  return (
    <SlidingPanel
      visible={visible}
      position={{ bottom: 72, left: 16, right: 16 }}
      axis="Y"
      zIndex={Z.sidePanel}
      className="mx-auto"
      style={{ maxWidth: 920 }}
    >
      <div className="glass-card relative">
        <PanelHeader
          onClose={onClose}
          actions={
            <button
              onClick={() => setShowCritical(v => !v)}
              className="text-[9px] font-mono px-1.5 py-0.5 rounded"
              style={{
                color: showCritical ? COLORS.holoHot : COLORS.textMuted,
                border: `1px solid ${showCritical ? COLORS.holoBase : COLORS.holoBorder06}`,
              }}
              title="Highlight the chain of work that set the run's duration"
            >
              critical path
            </button>
          }
        >
          <span className="text-[10px] font-mono tracking-wider" style={{ color: COLORS.textPrimary }}>
            EXECUTION TIMELINE
          </span>
          {total > 0 && (
            <span className="text-[9px] font-mono truncate" style={{ color: COLORS.textMuted }}>
              {formatSeconds(analysis.end - analysis.start)} · {analysis.lanes.length} agent{analysis.lanes.length === 1 ? '' : 's'} · critical path {toolShare}% tools, {100 - toolShare}% agents
            </span>
          )}
        </PanelHeader>

        <div className="overflow-auto relative" style={{ maxHeight: 320 }}>
          <canvas
            ref={canvasRef}
            style={{ display: 'block', cursor: dragging ? 'ew-resize' : onAgentClick ? 'pointer' : 'default' }}
            onMouseDown={e => {
              if (!onScrubHandle(e)) return
              e.preventDefault()
              setDragging(true)
              setHover(null)
              seekTo(e.clientX)
            }}
            onMouseMove={e => {
              if (dragging) return
              if (onScrubHandle(e)) {
                e.currentTarget.style.cursor = 'ew-resize'
                setHover(null)
                return
              }
              e.currentTarget.style.cursor = onAgentClick ? 'pointer' : 'default'
              const found = hitAt(e)
              setHover(found ? { x: found.x, y: found.y, text: found.hit.text } : null)
            }}
            onMouseLeave={() => setHover(null)}
            onClick={e => {
              if (onScrubHandle(e)) return
              const found = hitAt(e)
              if (found && onAgentClick) onAgentClick(found.hit.lane)
            }}
          />
          {hover && (
            <div
              className="absolute pointer-events-none text-[9px] font-mono px-2 py-1 rounded whitespace-pre"
              style={{
                left: Math.min(hover.x + 12, (canvasRef.current?.clientWidth ?? 600) - 220),
                top: hover.y + 14,
                maxWidth: 320,
                background: COLORS.cardBgDark,
                border: `1px solid ${COLORS.holoBorder06}`,
                color: COLORS.textPrimary,
                zIndex: 1,
              }}
            >
              {hover.text}
            </div>
          )}
        </div>

        {agents && <MomentStrip analysis={analysis} agents={agents} conversations={conversations} time={currentTime} onAgentClick={onAgentClick} />}

        {showCritical && analysis.topCritical.length > 0 && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 pt-1.5 text-[9px] font-mono" style={{ color: COLORS.textMuted }}>
            <span>Longest on the critical path:</span>
            {analysis.topCritical.map(item => (
              <button
                key={`${item.kind}-${item.lane}-${item.label}`}
                onClick={() => onAgentClick?.(item.lane)}
                className="hover:underline"
                style={{ color: item.kind === 'tool' ? COLORS.tool : COLORS.holoBright }}
                title={item.kind === 'tool' ? `${item.label} calls by ${item.lane}` : `${item.lane}'s own time: model calls and thinking`}
              >
                {item.kind === 'tool' ? `${item.label} (${item.lane})` : `${item.lane} itself`} {formatSeconds(item.duration)}
              </button>
            ))}
          </div>
        )}

        <div className="flex items-center gap-3 px-3 py-1.5" style={{ borderTop: `1px solid ${COLORS.holoBorder06}`, marginTop: 6 }}>
          {LEGEND_ITEMS.map(item => (
            <div key={item.label} className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm" style={{ background: item.color + '90' }} />
              <span className="text-[9px] font-mono" style={{ color: COLORS.textMuted }}>{item.label}</span>
            </div>
          ))}
          <span className="flex-1" />
          <span className="text-[9px] font-mono" style={{ color: COLORS.textFaint }}>{onSeek ? 'drag the ruler to scrub · ' : ''}hover for details · click to select</span>
        </div>
      </div>
    </SlidingPanel>
  )
}

// ─── The moment at the playhead ─────────────────────────────────────────────

const MOMENT_ROWS = 8

function activityOf(agent: Agent, messages: ConversationMessage[] | undefined): { text: string; color: string } {
  if (agent.state === 'waiting_permission') return { text: 'waiting for permission', color: COLORS.waiting_permission }
  if (agent.state === 'tool_calling' && agent.currentTool) return { text: `\u2699 ${agent.currentTool}`, color: COLORS.tool }
  const last = messages && [...messages].reverse().find(m => m.type === 'thinking' || m.type === 'assistant' || m.type === 'tool_result' || m.type === 'user')
  if (!last) return { text: agent.state, color: COLORS.textMuted }
  const text = last.content.replace(/\s+/g, ' ').trim()
  return {
    text: last.type === 'thinking' ? `thinking: ${text}` : last.type === 'tool_result' ? `${last.toolName ?? 'tool'} returned: ${text.replace(/^<\s*/, '')}` : text,
    color: last.type === 'thinking' ? COLORS.contextReasoning : COLORS.textDim,
  }
}

/** Each agent running at the playhead: how full its context was, what it
 *  held (when the runtime reports a breakdown), and what it was doing. */
function MomentStrip({ analysis, agents, conversations, time, onAgentClick }: {
  analysis: RunAnalysis
  agents: Map<string, Agent>
  conversations?: Map<string, ConversationMessage[]>
  time: number
  onAgentClick?: (agentId: string) => void
}) {
  const order = new Map(analysis.lanes.map((l, i) => [l.name, i]))
  const depth = new Map(analysis.lanes.map(l => [l.name, l.depth]))
  const live = [...agents.values()]
    .filter(a => a.state !== 'complete')
    // The main agent first (its name on screen can be the session label), then tree order
    .sort((a, b) => Number(b.isMain) - Number(a.isMain) || (order.get(a.name) ?? 1e9) - (order.get(b.name) ?? 1e9))
  if (!live.length) return null
  const shown = live.slice(0, MOMENT_ROWS)
  const elapsed = Math.max(0, time - analysis.start)

  return (
    <div className="px-3 pt-2 font-mono" style={{ borderTop: `1px solid ${COLORS.holoBorder06}`, marginTop: 4 }}>
      <div className="text-[9px] mb-1" style={{ color: COLORS.textMuted }}>
        At {formatSeconds(elapsed)}: {live.length} agent{live.length === 1 ? '' : 's'} running
      </div>
      <div className="flex flex-col gap-0.5">
        {shown.map(agent => {
          const max = agent.tokensMax || 1
          const usage = agent.tokensUsed / max
          const segments = contextSegments(agent.contextBreakdown).filter(s => s.value > 0)
          const classified = segments.reduce((sum, seg) => sum + seg.value, 0)
          const rest = Math.max(0, agent.tokensUsed - classified)
          const usageColor = usage > 0.9 ? COLORS.error : usage > 0.8 ? COLORS.tool : COLORS.holoBase
          const activity = activityOf(agent, conversations?.get(agent.name))
          const tint = modelColor(agent.model)
          return (
            <button
              key={agent.id}
              onClick={() => onAgentClick?.(agent.id)}
              className="grid items-center gap-2 text-[9px] text-left rounded px-1 hover:bg-white/5"
              style={{ gridTemplateColumns: `${LABEL_WIDTH - 12}px 120px 1fr` }}
              title={`${agent.name}${agent.model ? ` · ${agent.model}` : ''}\ncontext: ${agent.tokensUsed.toLocaleString()} / ${max.toLocaleString()} tokens (${Math.round(usage * 100)}%)\n${activity.text}`}
            >
              <span className="truncate flex items-center gap-1" style={{ color: COLORS.textDim, paddingLeft: Math.min(depth.get(agent.name) ?? 0, 6) * INDENT }}>
                <span className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: tint ?? COLORS.textMuted }} />
                <span className="truncate">{agent.name}</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="relative flex-1 h-[5px] rounded-sm overflow-hidden" style={{ background: COLORS.holoBg05 }}>
                  <span className="absolute inset-y-0 left-0 flex" style={{ width: `${Math.min(100, usage * 100)}%` }}>
                    {segments.map((seg, i) => <span key={i} style={{ width: `${(seg.value / Math.max(agent.tokensUsed, 1)) * 100}%`, background: seg.color }} />)}
                    {rest > 0 && <span style={{ width: `${(rest / Math.max(agent.tokensUsed, 1)) * 100}%`, background: usageColor }} />}
                  </span>
                </span>
                <span style={{ color: usage > 0.8 ? usageColor : COLORS.textMuted, minWidth: 30, textAlign: 'right' }}>{formatTokens(agent.tokensUsed)}</span>
              </span>
              <span className="truncate" style={{ color: activity.color }}>{activity.text}</span>
            </button>
          )
        })}
        {live.length > shown.length && (
          <span className="text-[9px] px-1" style={{ color: COLORS.textFaint }}>+{live.length - shown.length} more</span>
        )}
      </div>
    </div>
  )
}
