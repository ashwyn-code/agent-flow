'use client'

import { useRef, useEffect, useMemo, useState, useCallback } from 'react'
import { Z, type SimulationEvent } from '@/lib/agent-types'
import { COLORS } from '@/lib/colors'
import { analyzeRun, type RunAnalysis } from '@/lib/run-analysis'
import { PanelHeader, SlidingPanel } from './shared-ui'

interface TimelinePanelProps {
  visible: boolean
  events: readonly SimulationEvent[]
  currentTime: number
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

  // Playhead
  const px = xOf(Math.min(currentTime, a.end))
  ctx.fillStyle = COLORS.holoHot
  ctx.globalAlpha = 0.45
  ctx.fillRect(px, HEADER_HEIGHT, 1, height - HEADER_HEIGHT)
  ctx.globalAlpha = 1

  ctx.restore()
  return hits
}

// ─── Component ──────────────────────────────────────────────────────────────

export function TimelinePanel({ visible, events, currentTime, onAgentClick, onClose }: TimelinePanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const hitsRef = useRef<HitRect[]>([])
  const [showCritical, setShowCritical] = useState(true)
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null)

  const analysis = useMemo(() => (visible ? analyzeRun(events, currentTime) : null), [visible, events, currentTime])
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
            style={{ display: 'block', cursor: onAgentClick ? 'pointer' : 'default' }}
            onMouseMove={e => {
              const found = hitAt(e)
              setHover(found ? { x: found.x, y: found.y, text: found.hit.text } : null)
            }}
            onMouseLeave={() => setHover(null)}
            onClick={e => {
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
          <span className="text-[9px] font-mono" style={{ color: COLORS.textFaint }}>hover for details · click to select</span>
        </div>
      </div>
    </SlidingPanel>
  )
}
