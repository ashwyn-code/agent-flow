/**
 * Layered (top-down) layout for an agent's node graph, e.g. a LangGraph
 * StateGraph. Pure functions, no DOM: the Graph panel renders the result as SVG.
 *
 *  1. Back edges (loops) are found with a DFS from __start__ so the rest of
 *     the graph is acyclic.
 *  2. Ranks are longest-path depths over the forward edges; __end__ is pinned
 *     below everything else.
 *  3. Nodes within a rank are reordered by the barycenter of their neighbours
 *     (a few down/up sweeps) to reduce crossings.
 *  4. Forward edges are vertical béziers; loops are routed as arcs in lanes on
 *     the right so they read as "goes back up".
 */
import type { AgentGraph } from './agent-types'

export type LayoutEdgeKind = 'forward' | 'back' | 'self'

export interface LayoutNode {
  id: string
  x: number
  y: number
  w: number
  h: number
  rank: number
}

export interface LayoutEdge {
  id: string
  kind: LayoutEdgeKind
  /** SVG path data */
  path: string
  /** Where to put the traversal-count badge */
  labelX: number
  labelY: number
}

export interface GraphLayout {
  nodes: Map<string, LayoutNode>
  edges: Map<string, LayoutEdge>
  width: number
  height: number
}

const PAD = 14
const NODE_H = 26
const TERMINAL_H = 20
const RANK_GAP = 34
const NODE_GAP = 16
const LANE_GAP = 12
const CHAR_W = 6.2
const MIN_W = 56
const MAX_W = 160
const SWEEPS = 4

export function nodeSize(label: string, kind: string): { w: number; h: number } {
  if (kind === 'start' || kind === 'end') return { w: 56, h: TERMINAL_H }
  const w = Math.min(MAX_W, Math.max(MIN_W, Math.round(label.length * CHAR_W + 24)))
  return { w, h: NODE_H }
}

/** Edge ids that close a cycle (target is an ancestor on the DFS stack). */
export function findBackEdges(order: string[], edges: Array<{ id: string; source: string; target: string }>): Set<string> {
  const out = new Map<string, Array<{ id: string; target: string }>>()
  for (const id of order) out.set(id, [])
  for (const e of edges) out.get(e.source)?.push({ id: e.id, target: e.target })

  const back = new Set<string>()
  const color = new Map<string, 0 | 1 | 2>() // 0 new, 1 on stack, 2 done
  const roots = order.includes('__start__') ? ['__start__', ...order.filter(id => id !== '__start__')] : order

  for (const root of roots) {
    if (color.get(root)) continue
    // Iterative DFS keeping each frame's next-child index.
    const stack: Array<{ id: string; i: number }> = [{ id: root, i: 0 }]
    color.set(root, 1)
    while (stack.length) {
      const frame = stack[stack.length - 1]
      const children = out.get(frame.id) ?? []
      if (frame.i >= children.length) {
        color.set(frame.id, 2)
        stack.pop()
        continue
      }
      const { id: edgeId, target } = children[frame.i++]
      const c = color.get(target) ?? 0
      if (c === 1) back.add(edgeId)
      else if (c === 0) {
        color.set(target, 1)
        stack.push({ id: target, i: 0 })
      }
    }
  }
  return back
}

export function layoutGraph(graph: AgentGraph): GraphLayout {
  const order = graph.order.filter(id => graph.nodes[id])
  const edgeList = Object.values(graph.edges).filter(e => graph.nodes[e.source] && graph.nodes[e.target])
  const back = findBackEdges(order, edgeList)
  const forward = edgeList.filter(e => !back.has(e.id) && e.source !== e.target)

  // ─── Ranks (longest path over the forward DAG) ────────────────────────────
  const indeg = new Map(order.map(id => [id, 0]))
  const succ = new Map<string, string[]>(order.map(id => [id, []]))
  const pred = new Map<string, string[]>(order.map(id => [id, []]))
  for (const e of forward) {
    indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1)
    succ.get(e.source)!.push(e.target)
    pred.get(e.target)!.push(e.source)
  }
  const rank = new Map(order.map(id => [id, 0]))
  const queue = order.filter(id => indeg.get(id) === 0)
  while (queue.length) {
    const id = queue.shift()!
    for (const t of succ.get(id)!) {
      rank.set(t, Math.max(rank.get(t)!, rank.get(id)! + 1))
      indeg.set(t, indeg.get(t)! - 1)
      if (indeg.get(t) === 0) queue.push(t)
    }
  }
  const endIds = order.filter(id => graph.nodes[id].kind === 'end')
  const maxOther = Math.max(0, ...order.filter(id => graph.nodes[id].kind !== 'end').map(id => rank.get(id)!))
  for (const id of endIds) rank.set(id, maxOther + 1)
  for (const id of order) if (graph.nodes[id].kind === 'start') rank.set(id, 0)

  // ─── Order within ranks (barycenter sweeps) ───────────────────────────────
  const rankCount = Math.max(0, ...order.map(id => rank.get(id)!)) + 1
  const rows: string[][] = Array.from({ length: rankCount }, () => [])
  for (const id of order) rows[rank.get(id)!].push(id)
  const pos = new Map<string, number>()
  const index = () => rows.forEach(row => row.forEach((id, i) => pos.set(id, i)))
  index()
  const sweep = (r: number, neighbours: Map<string, string[]>) => {
    const row = rows[r]
    const bary = new Map(row.map((id, i) => {
      const ns = neighbours.get(id)!
      return [id, ns.length ? ns.reduce((sum, n) => sum + pos.get(n)!, 0) / ns.length : i]
    }))
    row.sort((a, b) => bary.get(a)! - bary.get(b)! || pos.get(a)! - pos.get(b)!)
    row.forEach((id, i) => pos.set(id, i))
  }
  for (let s = 0; s < SWEEPS; s++) {
    if (s % 2 === 0) for (let r = 1; r < rankCount; r++) sweep(r, pred)
    else for (let r = rankCount - 2; r >= 0; r--) sweep(r, succ)
  }

  // ─── Coordinates ──────────────────────────────────────────────────────────
  const size = new Map(order.map(id => [id, nodeSize(graph.nodes[id].label, graph.nodes[id].kind)]))
  const rowWidth = (row: string[]) => row.reduce((sum, id) => sum + size.get(id)!.w, 0) + NODE_GAP * Math.max(0, row.length - 1)
  const contentW = Math.max(0, ...rows.map(rowWidth))
  const nodes = new Map<string, LayoutNode>()
  let y = PAD
  rows.forEach((row, r) => {
    const rowH = Math.max(0, ...row.map(id => size.get(id)!.h))
    let x = PAD + (contentW - rowWidth(row)) / 2
    for (const id of row) {
      const { w, h } = size.get(id)!
      nodes.set(id, { id, x, y: y + (rowH - h) / 2, w, h, rank: r })
      x += w + NODE_GAP
    }
    y += rowH + RANK_GAP
  })
  const height = Math.max(PAD * 2, y - RANK_GAP + PAD)

  // ─── Loop lanes ───────────────────────────────────────────────────────────
  // Each back edge gets a lane on the side its source sits on (left of the
  // graph's centre loops left), so it never sweeps across a sibling node.
  // Shorter spans take inner lanes so arcs nest.
  const loops = edgeList
    .filter(e => back.has(e.id) && e.source !== e.target)
    .sort((a, b) => Math.abs(rank.get(a.source)! - rank.get(a.target)!) - Math.abs(rank.get(b.source)! - rank.get(b.target)!))
  const centre = PAD + contentW / 2
  const laneOf = new Map<string, { side: 'left' | 'right'; lane: number }>()
  let leftLanes = 0, rightLanes = 0
  for (const e of loops) {
    const s = nodes.get(e.source)!
    const side = s.x + s.w / 2 < centre - 1 ? 'left' : 'right'
    laneOf.set(e.id, { side, lane: side === 'left' ? leftLanes++ : rightLanes++ })
  }
  const shift = leftLanes * LANE_GAP
  if (shift) for (const n of nodes.values()) n.x += shift

  // ─── Edge paths ───────────────────────────────────────────────────────────
  const edges = new Map<string, LayoutEdge>()
  for (const e of forward) {
    const s = nodes.get(e.source)!, t = nodes.get(e.target)!
    const sx = s.x + s.w / 2, sy = s.y + s.h
    const tx = t.x + t.w / 2, ty = t.y
    const dy = Math.max(12, (ty - sy) / 2)
    edges.set(e.id, {
      id: e.id, kind: 'forward',
      path: `M ${sx} ${sy} C ${sx} ${sy + dy}, ${tx} ${ty - dy}, ${tx} ${ty}`,
      labelX: (sx + tx) / 2, labelY: (sy + ty) / 2,
    })
  }

  let rightmost = shift + PAD + contentW
  for (const e of loops) {
    const s = nodes.get(e.source)!, t = nodes.get(e.target)!
    const { side, lane } = laneOf.get(e.id)!
    const lo = Math.min(s.rank, t.rank), hi = Math.max(s.rank, t.rank)
    const span = [...nodes.values()].filter(n => n.rank >= lo && n.rank <= hi)
    const left = side === 'left'
    const laneX = left
      ? Math.min(...span.map(n => n.x)) - LANE_GAP * (lane + 1)
      : Math.max(...span.map(n => n.x + n.w)) + LANE_GAP * (lane + 1)
    const sx = left ? s.x : s.x + s.w, sy = s.y + s.h / 2
    const tx = left ? t.x : t.x + t.w, ty = t.y + t.h / 2
    // A cubic with both control points at cx peaks at ~0.75 of the way there.
    const cx = sx + (laneX - sx) * 4 / 3
    const cx2 = tx + (laneX - tx) * 4 / 3
    edges.set(e.id, {
      id: e.id, kind: 'back',
      path: `M ${sx} ${sy} C ${cx} ${sy}, ${cx2} ${ty}, ${tx} ${ty}`,
      labelX: laneX, labelY: (sy + ty) / 2,
    })
    if (!left) rightmost = Math.max(rightmost, laneX)
  }

  for (const e of edgeList) {
    if (e.source !== e.target) continue
    const n = nodes.get(e.source)!
    const x = n.x + n.w, y = n.y + n.h / 2
    edges.set(e.id, {
      id: e.id, kind: 'self',
      path: `M ${x} ${y - 6} C ${x + 26} ${y - 18}, ${x + 26} ${y + 18}, ${x} ${y + 6}`,
      labelX: x + 22, labelY: y,
    })
    rightmost = Math.max(rightmost, x + 22)
  }

  return { nodes, edges, width: rightmost + PAD + 10, height }
}
