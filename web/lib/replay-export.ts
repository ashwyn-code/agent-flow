/**
 * Replay export: one self-contained HTML file with the visualizer and a
 * session's events inside, that replays the run in any browser (no relay,
 * no network). Built from the page itself (Export replay in the canvas
 * menu) or from a JSONL event log (`pnpm replay:export`).
 *
 * The page carries its data in `window.__AGENT_FLOW_REPLAY__`; the bridge
 * sees it, skips the relay connection, and feeds the events in through the
 * normal live path at their original pace, so scrubbing, the timeline and
 * every panel work as they do live.
 */

export interface ReplayEvent {
  time: number
  type: string
  payload: Record<string, unknown>
}

export interface ReplayData {
  version: 1
  session: { id: string; label: string }
  events: ReplayEvent[]
  exportedAt: string
}

declare global {
  interface Window { __AGENT_FLOW_REPLAY__?: ReplayData }
}

/** Longest pause kept when replaying (seconds) */
export const MAX_REPLAY_GAP_S = 2

/** Data embedded in the current page, if it's an exported replay. */
export function embeddedReplay(): ReplayData | null {
  if (typeof window === 'undefined') return null
  const data = window.__AGENT_FLOW_REPLAY__
  return data && Array.isArray(data.events) && data.session ? data : null
}

/** JSON that's safe inside a <script> element. */
function scriptJson(value: unknown): string {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029')
}

/** Script text that can't close its <script> element early. */
function scriptText(js: string): string {
  return js.replace(/<\/(script)/gi, '<\\/$1')
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!))
}

/** Normalize events for export: times relative to the first, sorted. */
export function normalizeEvents(events: readonly { time: number; type: string; payload: unknown }[]): ReplayEvent[] {
  const valid = events.filter(e => typeof e.time === 'number' && Number.isFinite(e.time) && typeof e.type === 'string')
  if (!valid.length) return []
  const t0 = Math.min(...valid.map(e => e.time))
  return valid
    .map(e => ({ time: Math.round((e.time - t0) * 1000) / 1000, type: e.type, payload: (e.payload && typeof e.payload === 'object' ? e.payload : {}) as Record<string, unknown> }))
    .sort((a, b) => a.time - b.time)
}

export function buildReplayHtml(opts: { js: string; css: string; label: string; events: readonly { time: number; type: string; payload: unknown }[]; sessionId?: string; exportedAt?: Date }): string {
  const data: ReplayData = {
    version: 1,
    session: { id: opts.sessionId || 'replay', label: opts.label || 'Agent Flow replay' },
    events: normalizeEvents(opts.events),
    exportedAt: (opts.exportedAt ?? new Date()).toISOString(),
  }
  const title = `${data.session.label} · Agent Flow replay`
  return `<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="generator" content="Agent Flow replay export">
<title>${escapeHtml(title)}</title>
<style>html, body { height: 100%; margin: 0; padding: 0; }</style>
<style>${opts.css.replace(/<\/(style)/gi, '<\\/$1')}</style>
</head>
<body class="font-sans antialiased" style="background: #0a0a1a;">
<div id="root" style="height: 100%;"></div>
<div id="af-replay-note" style="position:fixed;right:16px;top:56px;max-width:calc(100vw - 32px);z-index:30;font:10px monospace;color:#aaeeffaa;background:rgba(5,5,16,.8);border:1px solid rgba(100,200,255,.12);border-radius:10px;padding:3px 10px;pointer-events:auto">
  Replay of <b style="color:#e6f7ff">${escapeHtml(data.session.label)}</b> · exported ${escapeHtml(data.exportedAt.slice(0, 16).replace('T', ' '))} UTC ·
  <a href="#" onclick="location.reload();return false" style="color:#66ccff">replay again</a>
</div>
<script>window.__AGENT_FLOW_REPLAY__ = ${scriptJson(data)};</script>
<script>${scriptText(opts.js)}</script>
</body>
</html>
`
}

/**
 * Feed an embedded replay into the page: the session, then its events at
 * their original pace (pauses shortened), as window messages the bridge
 * already understands. Returns a function that stops it.
 */
export function startEmbeddedReplay(data: ReplayData, post: (message: unknown) => void = m => window.postMessage(m, '*')): () => void {
  const { id, label } = data.session
  const now = Date.now()
  post({ type: 'session-list', sessions: [{ id, label, status: 'active', startTime: now, lastActivityTime: now }] })
  let i = 0
  let timer: ReturnType<typeof setTimeout> | null = null
  let stopped = false
  const step = () => {
    timer = null
    if (stopped) return
    const batch: (ReplayEvent & { sessionId: string })[] = []
    const t = data.events[i]?.time ?? 0
    while (i < data.events.length && data.events[i].time - t < 0.02) batch.push({ ...data.events[i++], sessionId: id })
    if (batch.length) post({ type: 'agent-event-batch', events: batch })
    if (i >= data.events.length) {
      post({ type: 'session-ended', sessionId: id })
      return
    }
    const gap = Math.min(Math.max(data.events[i].time - t, 0), MAX_REPLAY_GAP_S)
    timer = setTimeout(step, gap * 1000)
  }
  // Let the app mount and select the session first
  timer = setTimeout(step, 400)
  return () => { stopped = true; if (timer) clearTimeout(timer) }
}

/** Download a file from the page. */
export function downloadFile(name: string, content: string, type: string): void {
  const url = URL.createObjectURL(new Blob([content], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 5000)
}

/** A file name from a session label. */
export function replayFileName(label: string, ext: string): string {
  const slug = label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48) || 'session'
  return `agent-flow-${slug}.${ext}`
}

/** The page's own bundle and styles (single-file builds: the app and VS Code webview). */
export async function currentPageAssets(): Promise<{ js: string; css: string }> {
  // An exported replay has everything inline: the bundle is the largest script
  if (embeddedReplay()) {
    const inline = [...document.querySelectorAll<HTMLScriptElement>('script:not([src])')]
      .map(s => s.textContent ?? '')
      .filter(text => !text.startsWith('window.__AGENT_FLOW_REPLAY__'))
    const css = [...document.querySelectorAll('style')].map(s => s.textContent ?? '').sort((a, b) => b.length - a.length)[0] ?? ''
    return { js: inline.sort((a, b) => b.length - a.length)[0] ?? '', css }
  }
  const scripts = [...document.querySelectorAll<HTMLScriptElement>('script[src]')].filter(s => /index\.js(\?|$)/.test(s.src))
  const styles = [...document.querySelectorAll<HTMLLinkElement>('link[rel="stylesheet"]')].filter(l => /index\.css(\?|$)/.test(l.href))
  if (scripts.length !== 1) throw new Error('This page is not a single-bundle build (use the standalone app to export)')
  const [js, css] = await Promise.all([
    fetch(scripts[0].src).then(r => { if (!r.ok) throw new Error(`could not read ${scripts[0].src}`); return r.text() }),
    styles.length ? fetch(styles[0].href).then(r => (r.ok ? r.text() : '')) : Promise.resolve(''),
  ])
  return { js, css }
}
