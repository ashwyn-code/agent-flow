import { COLORS } from '@/lib/colors'
import { SPAWN_FX, COMPLETE_FX } from '@/lib/canvas-constants'
import { drawHexagon } from './draw-misc'
import { alphaHex } from '@/lib/utils'

export interface VisualEffect {
  type: 'spawn' | 'complete' | 'shatter' | 'compact' | 'error_ripple'
  x: number
  y: number
  color: string
  age: number
  duration: number
  particles?: Array<{ angle: number; speed: number; size: number }>
  /** Text shown with the effect (e.g. how much context was compacted) */
  label?: string
  /** Size the effect starts from (e.g. the agent's context ring) */
  radius?: number
}

export function drawEffects(ctx: CanvasRenderingContext2D, effects: VisualEffect[]) {
  for (const fx of effects) {
    const progress = fx.age / fx.duration
    ctx.save()

    switch (fx.type) {
      case 'spawn': {
        // Expanding hex ring + white flash
        const ringRadius = SPAWN_FX.ringStart + progress * SPAWN_FX.ringExpand
        const alpha = (1 - progress) * SPAWN_FX.maxAlpha

        // White flash (quick, first 30%)
        if (progress < SPAWN_FX.flashThreshold) {
          const flashAlpha = (1 - progress / SPAWN_FX.flashThreshold) * SPAWN_FX.flashAlpha
          ctx.beginPath()
          ctx.arc(fx.x, fx.y, SPAWN_FX.flashBaseRadius * (1 - progress / SPAWN_FX.flashThreshold) + SPAWN_FX.flashMinRadius, 0, Math.PI * 2)
          ctx.fillStyle = COLORS.holoHot + alphaHex(flashAlpha)
          ctx.fill()
        }

        // Expanding hexagonal ring
        ctx.globalAlpha = alpha
        drawHexagon(ctx, fx.x, fx.y, ringRadius)
        ctx.strokeStyle = fx.color
        ctx.lineWidth = 2 * (1 - progress)
        ctx.stroke()

        // Scatter particles outward
        for (let i = 0; i < SPAWN_FX.particleCount; i++) {
          const a = (i / SPAWN_FX.particleCount) * Math.PI * 2
          const d = ringRadius * 0.8 + progress * 20
          const px = fx.x + Math.cos(a) * d
          const py = fx.y + Math.sin(a) * d
          ctx.beginPath()
          ctx.fillStyle = fx.color + alphaHex(alpha * (200 / 255))
          ctx.arc(px, py, SPAWN_FX.particleSize * (1 - progress), 0, Math.PI * 2)
          ctx.fill()
        }
        break
      }

      case 'complete': {
        // White flash + expanding ring that fades
        const ringRadius = COMPLETE_FX.ringStart + progress * COMPLETE_FX.ringExpand
        const alpha = (1 - progress) * COMPLETE_FX.maxAlpha

        // Bright white flash (first 20%)
        if (progress < COMPLETE_FX.flashThreshold) {
          const flashAlpha = (1 - progress / COMPLETE_FX.flashThreshold) * COMPLETE_FX.flashAlpha
          const grad = ctx.createRadialGradient(fx.x, fx.y, 0, fx.x, fx.y, COMPLETE_FX.flashRadius)
          grad.addColorStop(0, COLORS.holoHot + alphaHex(flashAlpha))
          grad.addColorStop(1, COLORS.holoHot + '00')
          ctx.fillStyle = grad
          ctx.fillRect(fx.x - COMPLETE_FX.flashRadius, fx.y - COMPLETE_FX.flashRadius, COMPLETE_FX.flashRadius * 2, COMPLETE_FX.flashRadius * 2)
        }

        // Expanding ring
        ctx.globalAlpha = alpha
        ctx.beginPath()
        ctx.arc(fx.x, fx.y, ringRadius, 0, Math.PI * 2)
        ctx.strokeStyle = fx.color
        ctx.lineWidth = COMPLETE_FX.lineWidthMax * (1 - progress)
        ctx.stroke()

        // Glow behind ring
        const grad = ctx.createRadialGradient(fx.x, fx.y, ringRadius - COMPLETE_FX.glowInner, fx.x, fx.y, ringRadius + COMPLETE_FX.glowOuter)
        grad.addColorStop(0, fx.color + '00')
        grad.addColorStop(0.5, fx.color + alphaHex(alpha * (100 / 255)))
        grad.addColorStop(1, fx.color + '00')
        ctx.fillStyle = grad
        ctx.beginPath()
        ctx.arc(fx.x, fx.y, ringRadius + COMPLETE_FX.glowOuter, 0, Math.PI * 2)
        ctx.fill()
        break
      }

      case 'compact': {
        // Context shrank: the ring collapses inward, with a label that drifts up
        const r0 = fx.radius ?? 40
        const ease = 1 - Math.pow(1 - progress, 3)
        const ringR = r0 * (1 - 0.45 * ease)
        const alpha = 1 - progress
        ctx.globalAlpha = alpha
        ctx.setLineDash([3, 3])
        ctx.lineDashOffset = progress * 24
        ctx.beginPath()
        ctx.arc(fx.x, fx.y, ringR, 0, Math.PI * 2)
        ctx.strokeStyle = fx.color
        ctx.lineWidth = 2.5 * (1 - progress) + 0.5
        ctx.stroke()
        ctx.setLineDash([])
        for (let i = 0; i < 10; i++) {
          const a = (i / 10) * Math.PI * 2
          const d = r0 + 14 - ease * 16
          ctx.beginPath()
          ctx.fillStyle = fx.color
          ctx.arc(fx.x + Math.cos(a) * d, fx.y + Math.sin(a) * d, 1.6 * (1 - progress) + 0.4, 0, Math.PI * 2)
          ctx.fill()
        }
        if (fx.label) {
          ctx.font = '9px monospace'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'bottom'
          ctx.fillStyle = fx.color
          ctx.fillText(fx.label, fx.x, fx.y - r0 - 6 - ease * 14)
        }
        break
      }

      case 'error_ripple': {
        // A failure: two red shockwaves spreading from where it happened
        const r0 = fx.radius ?? 14
        for (let i = 0; i < 2; i++) {
          const p = Math.min(1, Math.max(0, (progress - i * 0.18) / 0.82))
          if (p <= 0) continue
          ctx.globalAlpha = (1 - p) * 0.75
          ctx.beginPath()
          ctx.arc(fx.x, fx.y, r0 + p * 46, 0, Math.PI * 2)
          ctx.strokeStyle = fx.color
          ctx.lineWidth = 2.2 * (1 - p) + 0.3
          ctx.shadowColor = fx.color
          ctx.shadowBlur = 8
          ctx.stroke()
        }
        break
      }

      case 'shatter': {
        // Particles scatter outward from tool card
        if (!fx.particles) break
        const alpha = (1 - progress) * 0.8
        ctx.globalAlpha = alpha

        for (const p of fx.particles) {
          const dist = p.speed * fx.age
          const px = fx.x + Math.cos(p.angle) * dist
          const py = fx.y + Math.sin(p.angle) * dist
          const size = p.size * (1 - progress * 0.7)

          ctx.beginPath()
          ctx.fillStyle = fx.color
          ctx.arc(px, py, size, 0, Math.PI * 2)
          ctx.fill()

          // Tiny glow on each particle
          ctx.beginPath()
          const glow = ctx.createRadialGradient(px, py, 0, px, py, size * 3)
          glow.addColorStop(0, fx.color + '40')
          glow.addColorStop(1, fx.color + '00')
          ctx.fillStyle = glow
          ctx.arc(px, py, size * 3, 0, Math.PI * 2)
          ctx.fill()
        }
        break
      }
    }

    ctx.restore()
  }
}
