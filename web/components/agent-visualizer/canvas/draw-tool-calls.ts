import { ToolCallNode } from '@/lib/agent-types'
import { COLORS, withAlpha } from '@/lib/colors'
import { TOOL_MAX_CARD_W, TOOL_DRAW } from '@/lib/canvas-constants'
import { truncateText } from './draw-misc'
import { measureTextCached } from './render-cache'

const GUARDRAIL = /^guardrail\b/i

/** Guardrail checks: a shield instead of a gear. */
export function isGuardrail(tool: ToolCallNode): boolean {
  return GUARDRAIL.test(tool.toolName)
}

function drawShield(ctx: CanvasRenderingContext2D, cx: number, cy: number, s: number, color: string, tripped: boolean) {
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(cx, cy - s)
  ctx.lineTo(cx + s * 0.85, cy - s * 0.6)
  ctx.lineTo(cx + s * 0.75, cy + s * 0.25)
  ctx.quadraticCurveTo(cx + s * 0.4, cy + s * 0.8, cx, cy + s)
  ctx.quadraticCurveTo(cx - s * 0.4, cy + s * 0.8, cx - s * 0.75, cy + s * 0.25)
  ctx.lineTo(cx - s * 0.85, cy - s * 0.6)
  ctx.closePath()
  ctx.fillStyle = color + '30'
  ctx.fill()
  ctx.strokeStyle = color
  ctx.lineWidth = 1.2
  ctx.stroke()
  ctx.beginPath()
  if (tripped) {
    // A cross: the check stopped the run
    ctx.moveTo(cx - s * 0.3, cy - s * 0.25); ctx.lineTo(cx + s * 0.3, cy + s * 0.35)
    ctx.moveTo(cx + s * 0.3, cy - s * 0.25); ctx.lineTo(cx - s * 0.3, cy + s * 0.35)
  } else {
    ctx.moveTo(cx - s * 0.35, cy + s * 0.05); ctx.lineTo(cx - s * 0.05, cy + s * 0.35); ctx.lineTo(cx + s * 0.4, cy - s * 0.3)
  }
  ctx.stroke()
  ctx.restore()
}

/** Dashed arcs from a failed call to its retry, labelled with the attempt. */
function drawRetryArcs(ctx: CanvasRenderingContext2D, toolCalls: Map<string, ToolCallNode>, time: number) {
  for (const tool of toolCalls.values()) {
    if (!tool.retryOf) continue
    const from = toolCalls.get(tool.retryOf)
    if (!from) continue
    const alpha = Math.min(tool.opacity, from.opacity)
    if (alpha <= 0.02) continue
    const dx = tool.x - from.x
    const dy = tool.y - from.y
    const len = Math.hypot(dx, dy) || 1
    // Bulge sideways so the arc doesn't run through the cards
    const bulge = Math.min(60, 18 + len * 0.35)
    const cx = (from.x + tool.x) / 2 - (dy / len) * bulge
    const cy = (from.y + tool.y) / 2 + (dx / len) * bulge
    ctx.save()
    ctx.globalAlpha = alpha * 0.85
    ctx.strokeStyle = COLORS.tool
    ctx.lineWidth = 1.3
    ctx.setLineDash([4, 3])
    ctx.lineDashOffset = -time * 14
    ctx.beginPath()
    ctx.moveTo(from.x, from.y)
    ctx.quadraticCurveTo(cx, cy, tool.x, tool.y)
    ctx.stroke()
    ctx.setLineDash([])
    // Label at the curve's midpoint
    const mx = 0.25 * from.x + 0.5 * cx + 0.25 * tool.x
    const my = 0.25 * from.y + 0.5 * cy + 0.25 * tool.y
    const text = `retry ${tool.attempt ?? 2}`
    ctx.font = `${TOOL_DRAW.tokenFontSize}px monospace`
    const w = measureTextCached(ctx, text) + 8
    ctx.fillStyle = COLORS.void
    ctx.beginPath()
    ctx.roundRect(mx - w / 2, my - 6, w, 12, 6)
    ctx.fill()
    ctx.strokeStyle = COLORS.tool + '90'
    ctx.lineWidth = 0.8
    ctx.stroke()
    ctx.fillStyle = COLORS.tool
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, mx, my + 0.5)
    ctx.restore()
  }
}

export function drawToolCalls(
  ctx: CanvasRenderingContext2D,
  toolCalls: Map<string, ToolCallNode>,
  time: number,
  selectedToolCallId?: string | null,
) {
  drawRetryArcs(ctx, toolCalls, time)
  for (const [id, tool] of toolCalls) {
    const isRunning = tool.state === 'running'
    const isError = tool.state === 'error'
    const pulse = isRunning ? Math.sin(time * 4) * 0.2 + 0.8 : isError ? Math.sin(time * 6) * 0.15 + 0.85 : 0.5

    ctx.save()
    ctx.globalAlpha = tool.opacity

    ctx.font = `${TOOL_DRAW.fontSize}px monospace`
    const guard = isGuardrail(tool)
    // The shield already says "guardrail"
    const name = guard ? tool.toolName.replace(/^guardrail:?\s*/i, '') || tool.toolName : tool.toolName
    const toolLabel = tool.args ? `${name}: ${tool.args}` : name
    const label = truncateText(ctx, toolLabel, TOOL_MAX_CARD_W - 12)
    const textWidth = Math.min(measureTextCached(ctx, label) + 12 + (guard ? 16 : 0), TOOL_MAX_CARD_W)
    const cardW = Math.max(60, textWidth)
    const cardH = (!isRunning && (tool.tokenCost || isError)) ? TOOL_DRAW.expandedHeight : TOOL_DRAW.collapsedHeight
    const cardX = tool.x - cardW / 2
    const cardY = tool.y - cardH / 2

    const isSelected = id === selectedToolCallId

    // Error glow
    if (isError) {
      ctx.shadowColor = COLORS.error
      ctx.shadowBlur = TOOL_DRAW.errorGlowBase + Math.sin(time * 6) * TOOL_DRAW.errorGlowPulse
    }

    ctx.beginPath()
    ctx.roundRect(cardX, cardY, cardW, cardH, TOOL_DRAW.borderRadius)
    ctx.fillStyle = isError
      ? withAlpha(COLORS.toolCardErrorBase, 0.8 * pulse)
      : isSelected ? withAlpha(COLORS.toolCardSelectedBase, 0.15 * pulse) : withAlpha(COLORS.toolCardBase, 0.7 * pulse)
    ctx.fill()
    ctx.strokeStyle = isError
      ? COLORS.error + '90'
      : isSelected ? COLORS.holoBase + 'aa' : isRunning ? COLORS.tool + '60' : COLORS.return + '40'
    ctx.lineWidth = isError ? 2 : isSelected ? 1.5 : 1
    ctx.stroke()

    ctx.shadowBlur = 0

    // Spinning ring
    if (isRunning) {
      ctx.beginPath()
      ctx.arc(tool.x, tool.y, Math.max(cardW, cardH) / 2 + TOOL_DRAW.spinRingPadding, time * TOOL_DRAW.spinSpeed, time * TOOL_DRAW.spinSpeed + TOOL_DRAW.spinArc)
      ctx.strokeStyle = COLORS.tool + '50'
      ctx.lineWidth = 1.5
      ctx.stroke()
    }

    // Crack lines for errors
    if (isError) {
      ctx.save()
      ctx.strokeStyle = COLORS.error + '40'
      ctx.lineWidth = 0.8
      for (let i = 0; i < 3; i++) {
        const a = (i / 3) * Math.PI * 2 + 0.5
        ctx.beginPath()
        ctx.moveTo(tool.x, tool.y)
        ctx.lineTo(tool.x + Math.cos(a) * cardW * 0.5, tool.y + Math.sin(a) * cardH * 0.6)
        ctx.stroke()
      }
      ctx.restore()
    }

    // Guardrails carry a shield: green when passed, red when tripped
    const textShift = guard ? 8 : 0
    if (guard) {
      const shieldColor = isError ? COLORS.error : isRunning ? COLORS.tool : COLORS.complete
      drawShield(ctx, cardX + 10, tool.y, 5.5, shieldColor, isError)
    }

    // Retries: which attempt this is
    if (tool.attempt && tool.attempt > 1) {
      const text = `\u21bb${tool.attempt}`
      ctx.font = `${TOOL_DRAW.tokenFontSize}px monospace`
      const bw = measureTextCached(ctx, text) + 7
      ctx.fillStyle = COLORS.void
      ctx.beginPath()
      ctx.roundRect(cardX + cardW - bw + 4, cardY - 6, bw, 11, 5.5)
      ctx.fill()
      ctx.strokeStyle = (isError ? COLORS.error : COLORS.tool) + 'b0'
      ctx.lineWidth = 0.8
      ctx.stroke()
      ctx.fillStyle = isError ? COLORS.error : COLORS.tool
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(text, cardX + cardW - bw / 2 + 4, cardY - 0.5)
    }

    ctx.font = `${TOOL_DRAW.fontSize}px monospace`
    const truncatedLabel = truncateText(ctx, toolLabel, cardW - 8 - textShift * 2)
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    const lx = tool.x + textShift

    if (isRunning) {
      ctx.fillStyle = COLORS.tool
      ctx.fillText(truncatedLabel, lx, tool.y)
    } else if (isError) {
      ctx.fillStyle = COLORS.error
      ctx.fillText(truncateText(ctx, `${name}: ${guard ? 'TRIPPED' : 'FAILED'}`, cardW - 8 - textShift * 2), lx, tool.y - TOOL_DRAW.twoLineOffset)
      ctx.font = `${TOOL_DRAW.errorFontSize}px monospace`
      ctx.fillStyle = COLORS.error + 'aa'
      ctx.fillText(truncateText(ctx, tool.errorMessage || tool.result || '', cardW - 8 - textShift * 2), lx, tool.y + TOOL_DRAW.twoLineOffset + 2)
    } else {
      // Completed card: show action + file path (most useful info at a glance)
      ctx.fillStyle = COLORS.return
      ctx.fillText(truncatedLabel, lx, tool.y - TOOL_DRAW.twoLineOffset)
      if (tool.tokenCost) {
        // Token cost as dim text below
        ctx.fillStyle = COLORS.tool + '90'
        ctx.font = `${TOOL_DRAW.tokenFontSize}px monospace`
        ctx.fillText(`${tool.tokenCost} tok`, tool.x, tool.y + TOOL_DRAW.twoLineOffset + 2)
      }
    }

    ctx.restore()
  }
}
