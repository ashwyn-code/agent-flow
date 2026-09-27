'use client'

import { memo, useMemo, useState } from 'react'
import { Z, type SimulationEvent } from '@/lib/agent-types'
import { COLORS } from '@/lib/colors'
import { modelColor, modelLabel } from '@/lib/model-colors'
import { analyzeRun } from '@/lib/run-analysis'
import { formatTokens } from '@/lib/utils'
import { agentCost } from './canvas/draw-cost'
import { stopPropagationHandlers } from './shared-ui'

export interface ModelUsage {
  model: string
  color: string
  agents: number
  tokens: number
  cost: number
}

/** Tokens and estimated cost per model for the whole run so far (finished
 *  agents included), most expensive first. */
export function modelUsage(events: readonly SimulationEvent[], currentTime: number): ModelUsage[] {
  const byModel = new Map<string, ModelUsage>()
  for (const lane of analyzeRun(events, currentTime).lanes) {
    if (!lane.model) continue
    const entry = byModel.get(lane.model) ?? { model: lane.model, color: modelColor(lane.model)!, agents: 0, tokens: 0, cost: 0 }
    entry.agents++
    entry.tokens += lane.tokens
    entry.cost += agentCost(lane.tokens, lane.model)
    byModel.set(lane.model, entry)
  }
  return [...byModel.values()].sort((a, b) => b.cost - a.cost || b.tokens - a.tokens)
}

function formatCost(cost: number): string {
  return cost < 0.01 ? `$${cost.toFixed(3)}` : `$${cost.toFixed(2)}`
}

interface ModelLegendProps {
  visible: boolean
  events: readonly SimulationEvent[]
  currentTime: number
  highlightModel: string | null
  onHighlight: (model: string | null) => void
}

/** Which model each agent runs on (the tint on its outer ring), with a
 *  running token and cost total per model. Hover a model to spotlight it. */
export const ModelLegend = memo(function ModelLegend({ visible, events, currentTime, highlightModel, onHighlight }: ModelLegendProps) {
  const [collapsed, setCollapsed] = useState(false)
  const usage = useMemo(() => (visible ? modelUsage(events, currentTime) : []), [visible, events, currentTime])
  if (!visible || usage.length === 0) return null
  const total = usage.reduce((sum, u) => sum + u.cost, 0)
  const maxCost = Math.max(...usage.map(u => u.cost), 1e-9)

  return (
    <div
      className="absolute glass-card font-mono"
      style={{ left: 12, bottom: 84, zIndex: Z.info, width: 'min(260px, calc(100vw - 24px))', padding: '6px 8px' }}
      onMouseLeave={() => onHighlight(null)}
      {...stopPropagationHandlers}
    >
      <button
        className="w-full flex items-center gap-2 text-[9px] tracking-wider"
        style={{ color: COLORS.textPrimary }}
        onClick={() => setCollapsed(c => !c)}
        aria-expanded={!collapsed}
        title={collapsed ? 'Show models' : 'Hide models'}
      >
        <span>MODELS</span>
        <span style={{ color: COLORS.textMuted }}>{usage.length}</span>
        <span className="flex-1" />
        <span style={{ color: COLORS.complete }}>~{formatCost(total)}</span>
        <span style={{ color: COLORS.textMuted }}>{collapsed ? '▸' : '▾'}</span>
      </button>
      {!collapsed && (
        <div className="mt-1.5 flex flex-col gap-1">
          {usage.map(u => {
            const active = highlightModel === u.model
            return (
              <div
                key={u.model}
                className="text-[9px] rounded px-1 py-0.5 cursor-default"
                style={{ background: active ? u.color + '18' : 'transparent', opacity: highlightModel && !active ? 0.5 : 1 }}
                onMouseEnter={() => onHighlight(u.model)}
                title={`${u.model}\n${u.agents} agent${u.agents === 1 ? '' : 's'} · ${u.tokens.toLocaleString()} tokens · ~${formatCost(u.cost)}`}
              >
                <div className="flex items-center gap-1.5">
                  <span className="inline-block w-2 h-2 rounded-full flex-shrink-0" style={{ background: u.color, boxShadow: `0 0 6px ${u.color}` }} />
                  <span className="truncate" style={{ color: u.color }}>{modelLabel(u.model)}</span>
                  <span className="flex-1" />
                  <span style={{ color: COLORS.textMuted }}>{u.agents}×</span>
                  <span style={{ color: COLORS.textDim, minWidth: 34, textAlign: 'right' }}>{formatTokens(u.tokens)}</span>
                  <span style={{ color: COLORS.complete, minWidth: 40, textAlign: 'right' }}>{formatCost(u.cost)}</span>
                </div>
                <div className="mt-0.5 h-[2px] rounded" style={{ background: COLORS.holoBg05 }}>
                  <div className="h-full rounded" style={{ width: `${Math.max(2, (u.cost / maxCost) * 100)}%`, background: u.color + 'b0' }} />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
})
