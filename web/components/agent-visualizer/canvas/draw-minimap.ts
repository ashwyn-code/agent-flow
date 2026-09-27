import type { Agent, Edge, ToolCallNode } from '@/lib/agent-types'
import { COLORS, getStateColor } from '@/lib/colors'
import type { CollapsedSummary } from '@/lib/collapse'
import { modelColor } from '@/lib/model-colors'

export const MINIMAP = {
  width: 176,
  height: 112,
  padding: 8,
  /** Shown automatically from this many agents, or when something is off screen */
  minAgents: 6,
} as const

interface Transform { x: number; y: number; scale: number }

/** Mapping between world coordinates and minimap pixels. */
export interface MinimapFrame {
  minX: number
  minY: number
  scale: number
  offsetX: number
  offsetY: number
}

export function minimapToWorld(frame: MinimapFrame, mx: number, my: number): { x: number; y: number } {
  return { x: (mx - frame.offsetX) / frame.scale + frame.minX, y: (my - frame.offsetY) / frame.scale + frame.minY }
}

function viewportWorld(t: Transform, w: number, h: number) {
  return { x0: -t.x / t.scale, y0: -t.y / t.scale, x1: (w - t.x) / t.scale, y1: (h - t.y) / t.scale }
}

/** Show the minimap for bigger runs, or whenever an agent is off screen. */
export function shouldShowMinimap(agents: Map<string, Agent>, t: Transform, w: number, h: number): boolean {
  if (agents.size === 0) return false
  if (agents.size >= MINIMAP.minAgents) return true
  const v = viewportWorld(t, w, h)
  for (const a of agents.values()) {
    if (a.x < v.x0 || a.x > v.x1 || a.y < v.y0 || a.y > v.y1) return true
  }
  return false
}

export function computeMinimapFrame(agents: Map<string, Agent>, toolCalls: Map<string, ToolCallNode>, t: Transform, w: number, h: number): MinimapFrame {
  const v = viewportWorld(t, w, h)
  let minX = v.x0, minY = v.y0, maxX = v.x1, maxY = v.y1
  for (const a of agents.values()) {
    minX = Math.min(minX, a.x - 30); maxX = Math.max(maxX, a.x + 30)
    minY = Math.min(minY, a.y - 30); maxY = Math.max(maxY, a.y + 30)
  }
  for (const tc of toolCalls.values()) {
    if (tc.opacity < 0.1) continue
    minX = Math.min(minX, tc.x - 40); maxX = Math.max(maxX, tc.x + 40)
    minY = Math.min(minY, tc.y - 12); maxY = Math.max(maxY, tc.y + 12)
  }
  const innerW = MINIMAP.width - MINIMAP.padding * 2
  const innerH = MINIMAP.height - MINIMAP.padding * 2
  const scale = Math.min(innerW / Math.max(maxX - minX, 1), innerH / Math.max(maxY - minY, 1))
  return {
    minX, minY, scale,
    offsetX: MINIMAP.padding + (innerW - (maxX - minX) * scale) / 2,
    offsetY: MINIMAP.padding + (innerH - (maxY - minY) * scale) / 2,
  }
}

export function drawMinimap(
  ctx: CanvasRenderingContext2D,
  frame: MinimapFrame,
  agents: Map<string, Agent>,
  toolCalls: Map<string, ToolCallNode>,
  edges: Edge[],
  collapsed: Map<string, CollapsedSummary>,
  t: Transform,
  w: number,
  h: number,
  time: number,
) {
  const px = (x: number) => frame.offsetX + (x - frame.minX) * frame.scale
  const py = (y: number) => frame.offsetY + (y - frame.minY) * frame.scale
  ctx.clearRect(0, 0, MINIMAP.width, MINIMAP.height)

  ctx.fillStyle = COLORS.cardBgDark
  ctx.beginPath()
  ctx.roundRect(0.5, 0.5, MINIMAP.width - 1, MINIMAP.height - 1, 8)
  ctx.fill()
  ctx.strokeStyle = COLORS.glassBorder
  ctx.lineWidth = 1
  ctx.stroke()

  // Parent-child links
  ctx.strokeStyle = COLORS.holoBase + '30'
  ctx.lineWidth = 0.8
  for (const e of edges) {
    if (e.type !== 'parent-child') continue
    const a = agents.get(e.from)
    const b = agents.get(e.to)
    if (!a || !b) continue
    ctx.beginPath()
    ctx.moveTo(px(a.x), py(a.y))
    ctx.lineTo(px(b.x), py(b.y))
    ctx.stroke()
  }

  // Tool calls
  for (const tc of toolCalls.values()) {
    if (tc.opacity < 0.1) continue
    ctx.fillStyle = (tc.state === 'error' ? COLORS.error : tc.state === 'running' ? COLORS.tool : COLORS.return) + 'a0'
    ctx.fillRect(px(tc.x) - 1.5, py(tc.y) - 1, 3, 2)
  }

  // Agents: model tint (or state), active ones glowing, collapsed ones ringed
  for (const [id, a] of agents) {
    if (a.opacity < 0.05) continue
    const x = px(a.x)
    const y = py(a.y)
    const r = a.isMain ? 3.5 : 2.4
    const active = a.state === 'thinking' || a.state === 'tool_calling' || a.state === 'waiting_permission'
    const color = a.state === 'error' ? COLORS.error : modelColor(a.model) ?? getStateColor(a.state)
    if (active) {
      ctx.fillStyle = color + '40'
      ctx.beginPath()
      ctx.arc(x, y, r + 2.5 + Math.sin(time * 4) * 0.8, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.fillStyle = a.state === 'complete' ? color + '70' : color
    ctx.beginPath()
    ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fill()
    const folded = collapsed.get(id)
    if (folded) {
      ctx.strokeStyle = folded.error ? COLORS.error : folded.active ? COLORS.tool : COLORS.holoBright
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(x, y, r + 2.2, 0, Math.PI * 2)
      ctx.stroke()
    }
  }

  // The part of the world on screen
  const v = viewportWorld(t, w, h)
  ctx.strokeStyle = COLORS.holoBright + 'c0'
  ctx.fillStyle = COLORS.holoBase + '12'
  ctx.lineWidth = 1
  const vx = px(v.x0)
  const vy = py(v.y0)
  ctx.fillRect(vx, vy, (v.x1 - v.x0) * frame.scale, (v.y1 - v.y0) * frame.scale)
  ctx.strokeRect(vx + 0.5, vy + 0.5, (v.x1 - v.x0) * frame.scale - 1, (v.y1 - v.y0) * frame.scale - 1)
}
