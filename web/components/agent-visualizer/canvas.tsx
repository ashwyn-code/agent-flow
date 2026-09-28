'use client'

import { useRef, useEffect, useState, useCallback } from 'react'
import { Agent, Particle, Edge, Discovery, DepthParticle } from '@/lib/agent-types'
import type { SimulationState } from '@/hooks/simulation/types'
import { getStateColor } from '@/lib/colors'
import { ANIM_SPEED, PERF_OVERLAY, PERF_OVERLAY_ENABLED } from '@/lib/canvas-constants'
import { BloomRenderer } from './bloom-renderer'
import { createDepthParticles, updateDepthParticles, drawBackground } from './background-layer'
import {
  type VisualEffect,
  drawTetherLine,
  drawEffects,
  drawAgents,
  drawMessageBubblesWorld,
  drawEdges, getActiveEdgeIds,
  drawParticles, buildEdgeMap,
  drawToolCalls,
  drawDiscoveries, drawDiscoveryConnections,
  drawCostLabels, drawCostSummaryPanel,
  detectStateChanges as detectStateChangesPure,
} from './canvas/index'
import { useCanvasCamera } from '@/hooks/use-canvas-camera'
import { applyCollapse, type VisibleScene } from '@/lib/collapse'
import { MINIMAP, computeMinimapFrame, drawMinimap, minimapToWorld, shouldShowMinimap, type MinimapFrame } from './canvas/draw-minimap'
import { useCanvasInteraction } from '@/hooks/use-canvas-interaction'

interface CanvasProps {
  /** Ref to simulation state — read every frame without React re-renders */
  simulationRef: React.RefObject<SimulationState>
  selectedAgentId: string | null
  hoveredAgentId: string | null
  showStats: boolean
  showHexGrid: boolean
  zoomToFitTrigger?: number
  pauseAutoFit?: boolean
  onAgentClick: (agentId: string | null) => void
  onAgentHover: (agentId: string | null) => void
  onAgentDrag: (agentId: string, x: number, y: number) => void
  onContextMenu: (e: React.MouseEvent, type: 'agent' | 'edge' | 'canvas', id?: string) => void
  onToolCallClick?: (toolCallId: string | null) => void
  selectedToolCallId?: string | null
  onDiscoveryClick?: (discoveryId: string | null) => void
  selectedDiscoveryId?: string | null
  showCostOverlay?: boolean
  /** Dim agents on other models (hovering the model legend) */
  highlightModel?: string | null
  /** Agents whose subtrees are folded away */
  collapsed?: ReadonlySet<string>
  /** Minimap: shown for bigger runs ('auto'), always, or never */
  minimap?: 'auto' | 'on' | 'off'
}

export function AgentCanvas({
  simulationRef,
  selectedAgentId, hoveredAgentId, showStats, showHexGrid, zoomToFitTrigger, pauseAutoFit,
  onAgentClick, onAgentHover, onAgentDrag, onContextMenu, onToolCallClick, selectedToolCallId, onDiscoveryClick, selectedDiscoveryId, showCostOverlay, highlightModel, collapsed, minimap = 'auto',
}: CanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mainCanvasRef = useRef<HTMLCanvasElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const animationRef = useRef<number>(0)
  const timeRef = useRef(0)
  const simTimeRef = useRef(0)
  const bloomRef = useRef<BloomRenderer | null>(null)
  const depthParticlesRef = useRef<DepthParticle[]>([])
  const lastFrameTimeRef = useRef(0)
  const dprRef = useRef(1)

  // Effects system
  const effectsRef = useRef<VisualEffect[]>([])
  const prevAgentStatesRef = useRef<Map<string, string>>(new Map())
  const prevToolStatesRef = useRef<Map<string, string>>(new Map())
  const prevTokensRef = useRef<Map<string, number>>(new Map())

  // Collapsed subtrees: the visible scene, cached while its inputs are unchanged
  const collapseCacheRef = useRef<{ key: unknown[]; scene: VisibleScene } | null>(null)
  const minimapRef = useRef<HTMLCanvasElement>(null)
  const minimapFrameRef = useRef<MinimapFrame | null>(null)

  // Rate-limited error logging for the draw loop (avoid flooding console)
  const lastDrawErrorRef = useRef(0)

  // Performance overlay state
  const perfRef = useRef({
    frames: 0,
    lastFpsUpdate: 0,
    fps: 0,
    frameTimeMs: 0,
    frameTimes: [] as number[],
    p95: 0,
  })

  // Caches for per-frame lookups — avoid rebuilding Set/Map every ~16ms
  const edgeLookupCacheRef = useRef<{
    particles: Particle[]
    edges: Edge[]
    activeEdgeIds: Set<string>
    edgeMap: Map<string, Edge>
  }>({ particles: [], edges: [], activeEdgeIds: new Set(), edgeMap: new Map() })

  // ─── Stable refs for animation loop & event handlers ────────────────────
  // Simulation data (agents, particles, etc.) is synced from simulationRef
  // at the top of each draw frame, so it's always fresh even without re-renders.
  const sim = simulationRef.current
  const makeDrawProps = (prev?: { isDragging: boolean }) => ({
    agents: sim.agents, toolCalls: sim.toolCalls,
    particles: sim.particles, edges: sim.edges, discoveries: sim.discoveries,
    selectedAgentId, hoveredAgentId, showStats, showHexGrid,
    showCostOverlay, selectedToolCallId, selectedDiscoveryId, highlightModel, collapsed, minimap,
    collapsedSummaries: new Map() as VisibleScene['collapsed'],
    simTime: sim.currentTime, pauseAutoFit, dimensions,
    onAgentDrag, onAgentClick, onAgentHover, onContextMenu,
    onToolCallClick, onDiscoveryClick,
    isDragging: prev?.isDragging ?? false,
  })
  const drawPropsRef = useRef(makeDrawProps())
  drawPropsRef.current = makeDrawProps(drawPropsRef.current)

  // ─── Camera ─────────────────────────────────────────────────────────────
  const {
    transformRef, userHasNavigatedRef, panVelocityRef, cancelCameraTarget,
    screenToCanvas, doZoomToFit, updateCamera,
  } = useCanvasCamera({
    mainCanvasRef, drawPropsRef, simTimeRef, dimensions,
    agentCount: sim.agents.size, zoomToFitTrigger, selectedAgentId,
  })

  // ─── Interaction ────────────────────────────────────────────────────────
  const {
    isDragging, handlers, updateDragLerp,
  } = useCanvasInteraction({
    drawPropsRef, transformRef, userHasNavigatedRef, panVelocityRef,
    simTimeRef, screenToCanvas, doZoomToFit, mainCanvasRef,
  })

  // Keep drawPropsRef in sync with interaction state
  drawPropsRef.current.isDragging = isDragging

  // ─── Setup ──────────────────────────────────────────────────────────────

  useEffect(() => {
    bloomRef.current = new BloomRenderer(0.5)
    depthParticlesRef.current = createDepthParticles(dimensions.width, dimensions.height)
    return () => { bloomRef.current = null }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- particles created once, resized by draw loop
  }, [])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const dpr = window.devicePixelRatio || 1
    dprRef.current = dpr
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width
        const h = entry.contentRect.height
        setDimensions({ width: w, height: h })
        bloomRef.current?.resize(w * dpr, h * dpr)
      }
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  // ─── Detect state changes → spawn effects ──────────────────────────────

  const detectStateChanges = useCallback(() => {
    const { agents, toolCalls } = drawPropsRef.current
    const { effects, newAgentStates, newToolStates, newTokens } = detectStateChangesPure(
      agents, toolCalls,
      prevAgentStatesRef.current, prevToolStatesRef.current, prevTokensRef.current,
    )
    effectsRef.current.push(...effects)
    prevAgentStatesRef.current = newAgentStates
    prevToolStatesRef.current = newToolStates
    prevTokensRef.current = newTokens
  }, [])

  // ─── Main draw loop ────────────────────────────────────────────────────

  // Stable ref so the rAF loop always calls the latest draw without
  // re-subscribing when the callback identity changes.
  const drawRef = useRef<(timestamp: number) => void>(() => {})

  const draw = useCallback((timestamp: number) => {
    animationRef.current = requestAnimationFrame((ts) => drawRef.current(ts))

    const canvas = mainCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    try {
      // Sync simulation data from ref — always fresh, independent of React renders
      {
        const s = simulationRef.current
        const p = drawPropsRef.current
        const folded = p.collapsed
        if (folded && folded.size > 0) {
          // Leave collapsed subtrees out of drawing, fitting and hit-testing
          const key = [s.agents, s.toolCalls, s.particles, s.edges, s.discoveries, folded]
          const cache = collapseCacheRef.current
          const scene = cache && cache.key.every((k, i) => k === key[i])
            ? cache.scene
            : applyCollapse({ agents: s.agents, toolCalls: s.toolCalls, edges: s.edges, particles: s.particles, discoveries: s.discoveries }, folded)
          collapseCacheRef.current = { key, scene }
          p.agents = scene.agents
          p.toolCalls = scene.toolCalls
          p.particles = scene.particles
          p.edges = scene.edges
          p.discoveries = scene.discoveries
          p.collapsedSummaries = scene.collapsed
        } else {
          p.agents = s.agents
          p.toolCalls = s.toolCalls
          p.particles = s.particles
          p.edges = s.edges
          p.discoveries = s.discoveries
          if (p.collapsedSummaries.size) p.collapsedSummaries = new Map()
        }
        p.simTime = s.currentTime
      }

      const {
        agents, toolCalls, particles, edges, discoveries,
        selectedAgentId, hoveredAgentId, showStats, showHexGrid,
        showCostOverlay, selectedToolCallId, selectedDiscoveryId, highlightModel, collapsedSummaries, minimap,
        simTime, pauseAutoFit, dimensions, onAgentDrag,
        isDragging,
      } = drawPropsRef.current
      const transform = transformRef.current

      const deltaTime = lastFrameTimeRef.current ? (timestamp - lastFrameTimeRef.current) / 1000 : ANIM_SPEED.defaultDeltaTime
      lastFrameTimeRef.current = timestamp
      timeRef.current += deltaTime
      if (simTime != null) simTimeRef.current = simTime

      const dpr = dprRef.current
      const w = dimensions.width
      const h = dimensions.height

      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr
        canvas.height = h * dpr
        ctx.scale(dpr, dpr)
      }

      // Camera physics (inertia + auto-fit)
      updateCamera(isDragging, pauseAutoFit)

      // Floaty agent drag
      updateDragLerp(agents, onAgentDrag)

      // Detect state changes → visual effects
      detectStateChanges()

      // Update effects (mutate in place to avoid GC pressure)
      {
        const effects = effectsRef.current
        let writeIdx = 0
        for (let i = 0; i < effects.length; i++) {
          effects[i].age += deltaTime
          if (effects[i].age < effects[i].duration) {
            if (writeIdx !== i) effects[writeIdx] = effects[i]
            writeIdx++
          }
        }
        effects.length = writeIdx
      }

      ctx.clearRect(0, 0, w, h)
      updateDepthParticles(depthParticlesRef.current, deltaTime, w, h)

      let activeAgentPos: { x: number; y: number; color: string } | undefined
      for (const [, agent] of agents) {
        if (agent.state === 'thinking' || agent.state === 'tool_calling' || agent.state === 'waiting_permission') {
          activeAgentPos = { x: agent.x, y: agent.y, color: getStateColor(agent.state) }
          break
        }
      }

      drawBackground(ctx, w, h, depthParticlesRef.current, transform, showHexGrid, timeRef.current, activeAgentPos)

      ctx.save()
      ctx.translate(transform.x, transform.y)
      ctx.scale(transform.scale, transform.scale)

      // Pre-compute shared lookup structures — cached across frames when inputs are unchanged
      const elCache = edgeLookupCacheRef.current
      let activeEdgeIds: Set<string>
      let edgeMap: Map<string, Edge>
      if (elCache.particles === particles && elCache.edges === edges) {
        activeEdgeIds = elCache.activeEdgeIds
        edgeMap = elCache.edgeMap
      } else {
        activeEdgeIds = getActiveEdgeIds(particles)
        edgeMap = buildEdgeMap(edges)
        edgeLookupCacheRef.current = { particles, edges, activeEdgeIds, edgeMap }
      }

      drawDiscoveryConnections(ctx, discoveries, agents)
      drawEdges(ctx, edges, agents, toolCalls, activeEdgeIds, timeRef.current)
      drawToolCalls(ctx, toolCalls, timeRef.current, selectedToolCallId)
      drawDiscoveries(ctx, discoveries, agents, selectedDiscoveryId)
      drawAgents(ctx, agents, selectedAgentId, hoveredAgentId, showStats, timeRef.current, highlightModel, collapsedSummaries)
      drawMessageBubblesWorld(ctx, agents, simTimeRef.current)
      if (showCostOverlay) drawCostLabels(ctx, agents, toolCalls)
      drawParticles(ctx, particles, edgeMap, agents, toolCalls, timeRef.current)
      drawEffects(ctx, effectsRef.current)

      if (selectedAgentId) {
        const agent = agents.get(selectedAgentId)
        if (agent) drawTetherLine(ctx, agent, transform, h)
      }

      ctx.restore()

      if (showCostOverlay) drawCostSummaryPanel(ctx, agents, toolCalls)
      if (bloomRef.current) bloomRef.current.apply(canvas, ctx)

      // Minimap (its own small canvas, drawn in the same frame)
      const mini = minimapRef.current
      if (mini) {
        const show = minimap === 'on' || (minimap === 'auto' && shouldShowMinimap(agents, transformRef.current, w, h))
        if (mini.style.display !== (show ? 'block' : 'none')) mini.style.display = show ? 'block' : 'none'
        if (show) {
          const mctx = mini.getContext('2d')
          if (mctx) {
            if (mini.width !== MINIMAP.width * dpr) {
              mini.width = MINIMAP.width * dpr
              mini.height = MINIMAP.height * dpr
            }
            mctx.setTransform(dpr, 0, 0, dpr, 0, 0)
            const frame = computeMinimapFrame(agents, toolCalls, transformRef.current, w, h)
            minimapFrameRef.current = frame
            drawMinimap(mctx, frame, agents, toolCalls, edges, collapsedSummaries, transformRef.current, w, h, timeRef.current)
          }
        }
      }

      // ─── Performance overlay (enabled via ?perf or ?stress) ──────────
      if (PERF_OVERLAY_ENABLED) {
        const perf = perfRef.current
        const frameEnd = performance.now()
        const frameMs = frameEnd - (timestamp || frameEnd)
        perf.frameTimes.push(frameMs)
        if (perf.frameTimes.length > PERF_OVERLAY.maxFrameSamples) perf.frameTimes.shift()
        perf.frames++
        perf.frameTimeMs = frameMs
        if (frameEnd - perf.lastFpsUpdate >= PERF_OVERLAY.updateIntervalMs) {
          perf.fps = perf.frames
          perf.frames = 0
          perf.lastFpsUpdate = frameEnd
          const sorted = [...perf.frameTimes].sort((a, b) => a - b)
          perf.p95 = sorted[Math.floor(sorted.length * 0.95)] || 0
        }
        const po = PERF_OVERLAY
        const textX = po.x + po.padding
        let textY = po.y + po.lineHeight + 2
        ctx.save()
        ctx.fillStyle = po.bgColor
        ctx.fillRect(po.x, po.y, po.width, po.height)
        ctx.font = po.font
        ctx.fillStyle = perf.fps < po.fpsWarning ? po.fpsWarningColor : perf.fps < po.fpsCaution ? po.fpsCautionColor : po.fpsGoodColor
        ctx.fillText(`FPS: ${perf.fps}`, textX, textY); textY += po.lineHeight
        ctx.fillStyle = po.textColor
        ctx.fillText(`Frame: ${frameMs.toFixed(1)}ms  P95: ${perf.p95.toFixed(1)}ms`, textX, textY); textY += po.lineHeight
        ctx.fillText(`Agents: ${agents.size}`, textX, textY); textY += po.lineHeight
        ctx.fillText(`Tool calls: ${toolCalls.size}`, textX, textY); textY += po.lineHeight
        ctx.fillText(`Particles: ${particles.length}`, textX, textY); textY += po.lineHeight
        ctx.fillText(`Edges: ${edges.length}`, textX, textY); textY += po.lineHeight
        ctx.fillText(`Discoveries: ${discoveries.length}`, textX, textY)
        ctx.restore()
      }

    } catch (err) {
      // Log at most once every 5s to avoid flooding the console
      const now = Date.now()
      if (now - lastDrawErrorRef.current > 5000) {
        lastDrawErrorRef.current = now
        console.warn('[AgentCanvas] draw error:', err)
      }
    }
  }, [detectStateChanges, updateCamera, updateDragLerp, transformRef])

  drawRef.current = draw

  /** Center the view on the minimap point under the pointer. */
  const jumpTo = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const frame = minimapFrameRef.current
    if (!frame) return
    const rect = e.currentTarget.getBoundingClientRect()
    const world = minimapToWorld(frame, e.clientX - rect.left, e.clientY - rect.top)
    const t = transformRef.current
    const { width, height } = drawPropsRef.current.dimensions
    userHasNavigatedRef.current = true
    panVelocityRef.current.active = false
    cancelCameraTarget()
    transformRef.current = { x: width / 2 - world.x * t.scale, y: height / 2 - world.y * t.scale, scale: t.scale }
  }, [transformRef, userHasNavigatedRef, panVelocityRef, cancelCameraTarget])

  useEffect(() => {
    const loop = (timestamp: number) => drawRef.current(timestamp)
    animationRef.current = requestAnimationFrame(loop)
    return () => { if (animationRef.current) cancelAnimationFrame(animationRef.current) }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- drawRef is stable; rAF loop set up once
  }, [])

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden" style={{ cursor: isDragging ? 'grabbing' : 'grab' }}>
      <canvas
        ref={mainCanvasRef}
        style={{ width: dimensions.width, height: dimensions.height }}
        {...handlers}
        className="w-full h-full"
      />
      <canvas
        ref={minimapRef}
        className="absolute"
        style={{ right: 12, bottom: 84, width: MINIMAP.width, height: MINIMAP.height, display: 'none', cursor: 'crosshair', zIndex: 5 }}
        aria-label="Minimap: click or drag to move the view"
        onMouseDown={e => { e.stopPropagation(); jumpTo(e) }}
        onMouseMove={e => { if (e.buttons & 1) jumpTo(e) }}
        onWheel={e => e.stopPropagation()}
        onContextMenu={e => e.stopPropagation()}
      />
    </div>
  )
}
