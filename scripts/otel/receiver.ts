/**
 * OpenTelemetry traces → relay sessions.
 *
 * `POST /v1/traces` is an OTLP/HTTP trace receiver (protobuf or JSON, plain
 * or gzip), so any OpenTelemetry SDK or Collector can export to the relay:
 *
 *   OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:3001/v1/traces
 *
 * Spans arrive as they end, children before parents, so each trace is held
 * until its root span arrives (or it goes quiet), converted, and then
 * replayed into its own session at the pace it originally ran, with long
 * pauses shortened.
 *
 * Access follows `/ingest`: loopback only, unless an ingest token is set.
 */
import * as crypto from 'crypto'
import * as fs from 'fs'
import type * as http from 'http'
import * as path from 'path'
import * as zlib from 'zlib'

import type { AgentEvent, SessionInfo } from '../../extension/src/protocol'
import { readNewFileLines } from '../../extension/src/fs-utils'
import { SessionTracker, type EventLogCallbacks } from '../event-log-watcher'
import { authorize, readBody, send } from '../ingest'
import { collectDeclaredGraphs, convertTrace, groupByTrace, type GraphRegistry } from './convert'
import { spansFromJson, spansFromProtobuf, spansFromText, type Span } from './otlp'

/** Compressed or plain request body */
export const MAX_OTLP_BYTES = 16 * 1024 * 1024
/** After decompression */
const MAX_OTLP_INFLATED_BYTES = 64 * 1024 * 1024
const MAX_PENDING_TRACES = 1000
const MAX_SPANS_PER_TRACE = 20000
const MAX_SESSIONS = 200
const MAX_DONE_REMEMBERED = 5000

export interface OtlpReceiverOptions {
  token?: string
  /** Show a trace this long after its last span if its root never arrives
   *  (e.g. the root is in an upstream service that doesn't export here). */
  idleMs?: number
  /** After the root span arrives, wait this long for stragglers. */
  settleMs?: number
  /** Longest pause kept when replaying a trace, in seconds. */
  maxGapS?: number
  /** Replay speed multiplier; 0 emits a trace's events all at once. */
  speed?: number
}

interface PendingTrace {
  spans: Span[]
  hasRoot: boolean
  timer: NodeJS.Timeout | null
}

export class OtlpReceiver {
  private readonly pending = new Map<string, PendingTrace>()
  private readonly done = new Set<string>()
  private readonly trackers = new Map<string, SessionTracker>()
  private readonly registry: GraphRegistry = new Map()
  private readonly timers = new Set<NodeJS.Timeout>()
  private readonly idleMs: number
  private readonly settleMs: number
  private readonly maxGapS: number
  private readonly speed: number
  /** Spans that arrived after their trace was already shown */
  lateSpans = 0
  /** Spans dropped because of the buffer limits */
  droppedSpans = 0

  constructor(private readonly callbacks: EventLogCallbacks, private readonly options: OtlpReceiverOptions = {}) {
    this.idleMs = options.idleMs ?? 15_000
    this.settleMs = options.settleMs ?? 750
    this.maxGapS = options.maxGapS ?? 2
    this.speed = options.speed ?? 1
  }

  getSessions(): SessionInfo[] {
    return [...this.trackers.values()].flatMap(t => t.getSessions())
  }

  dispose(): void {
    for (const timer of this.timers) clearTimeout(timer)
    this.timers.clear()
    for (const trace of this.pending.values()) if (trace.timer) clearTimeout(trace.timer)
    this.pending.clear()
  }

  /** `POST /v1/traces` */
  handle(req: http.IncomingMessage, res: http.ServerResponse): void {
    if (!authorize(req, res, this.options.token)) return
    readBody(req, res, MAX_OTLP_BYTES, raw => {
      const encoding = String(req.headers['content-encoding'] ?? '').toLowerCase()
      const json = String(req.headers['content-type'] ?? '').toLowerCase().includes('json')
      let body: Buffer
      try {
        body = encoding === 'gzip' ? zlib.gunzipSync(raw, { maxOutputLength: MAX_OTLP_INFLATED_BYTES })
          : encoding === 'deflate' ? zlib.inflateSync(raw, { maxOutputLength: MAX_OTLP_INFLATED_BYTES })
          : raw
      } catch {
        return send(res, 400, { error: 'could not decompress body' })
      }
      let spans: Span[]
      try {
        spans = json ? spansFromJson(JSON.parse(body.toString('utf8'))) : spansFromProtobuf(body)
      } catch {
        return send(res, 400, { error: `body is not an OTLP ${json ? 'JSON' : 'protobuf'} ExportTraceServiceRequest` })
      }
      this.addSpans(spans)
      if (json) return send(res, 200, {})
      // An empty ExportTraceServiceResponse
      res.writeHead(200, { 'Content-Type': 'application/x-protobuf' })
      res.end()
    })
  }

  /** Buffer spans until their traces are complete. */
  addSpans(spans: Span[]): void {
    collectDeclaredGraphs(spans, this.registry)
    for (const [traceId, traceSpans] of groupByTrace(spans)) {
      if (this.done.has(traceId)) {
        this.lateSpans += traceSpans.length
        continue
      }
      let trace = this.pending.get(traceId)
      if (!trace) {
        if (this.pending.size >= MAX_PENDING_TRACES) {
          // Show the oldest waiting trace now rather than dropping new ones
          const oldest = this.pending.keys().next().value
          if (oldest !== undefined) this.complete(oldest)
        }
        trace = { spans: [], hasRoot: false, timer: null }
        this.pending.set(traceId, trace)
      }
      const room = MAX_SPANS_PER_TRACE - trace.spans.length
      if (traceSpans.length > room) this.droppedSpans += traceSpans.length - Math.max(0, room)
      trace.spans.push(...traceSpans.slice(0, Math.max(0, room)))
      if (traceSpans.some(s => !s.parentSpanId)) trace.hasRoot = true
      if (trace.timer) clearTimeout(trace.timer)
      trace.timer = setTimeout(() => this.complete(traceId), trace.hasRoot ? this.settleMs : this.idleMs)
    }
  }

  /** Convert and show every buffered trace now (e.g. at the end of a file). */
  flush(): void {
    for (const traceId of [...this.pending.keys()]) this.complete(traceId)
  }

  private complete(traceId: string): void {
    const trace = this.pending.get(traceId)
    if (!trace) return
    if (trace.timer) clearTimeout(trace.timer)
    this.pending.delete(traceId)
    this.done.add(traceId)
    if (this.done.size > MAX_DONE_REMEMBERED) this.done.delete(this.done.values().next().value as string)

    let converted
    try {
      converted = convertTrace(trace.spans, { registry: this.registry })
    } catch (err) {
      console.error(`[otel] could not convert trace ${traceId}:`, err)
      return
    }
    if (!converted || !converted.events.length) return

    const hash = crypto.createHash('sha256').update(traceId).digest('hex').slice(0, 8)
    const tracker = new SessionTracker(`otel-${hash}`, converted.name, this.callbacks)
    this.trackers.set(traceId, tracker)
    if (this.trackers.size > MAX_SESSIONS) this.trackers.delete(this.trackers.keys().next().value as string)
    this.replay(tracker, converted.events)
  }

  private replay(tracker: SessionTracker, events: AgentEvent[]): void {
    if (this.speed <= 0) {
      for (const event of events) tracker.handle(event)
      return
    }
    let i = 0
    const next = () => {
      while (i < events.length) {
        const event = events[i++]
        tracker.handle(event)
        if (i >= events.length) return
        const gap = Math.min(Math.max(events[i].time - event.time, 0), this.maxGapS)
        if (gap > 0) {
          const timer = setTimeout(() => { this.timers.delete(timer); next() }, (gap * 1000) / this.speed)
          this.timers.add(timer)
          return
        }
      }
    }
    next()
  }
}

/**
 * Follows an OTLP JSON file — one ExportTraceServiceRequest, or JSON Lines of
 * them as the collector's file exporter writes — and feeds it to a receiver.
 */
export class OtlpFileWatcher {
  private fileSize = 0
  private tail = ''
  /** Lines of a pretty-printed (multi-line) document read so far */
  private doc: string[] = []
  private pollTimer: NodeJS.Timeout | null = null

  constructor(readonly filePath: string, private readonly receiver: OtlpReceiver, private readonly pollMs = 500) {}

  start(): void {
    this.poll()
    this.pollTimer = setInterval(() => this.poll(), this.pollMs)
  }

  dispose(): void {
    if (this.pollTimer) clearInterval(this.pollTimer)
    this.pollTimer = null
  }

  /** Exposed for tests; normally driven by the poll timer. */
  poll(): void {
    const result = readNewFileLines(this.filePath, this.fileSize, this.tail)
    if (!result) return this.tryTail()
    if (result.newSize === 0 && this.fileSize > 0) {
      // Truncated: start over
      this.fileSize = 0
      this.tail = ''
      this.doc = []
      return
    }
    this.fileSize = result.newSize
    this.tail = result.tail
    for (const line of result.lines) this.line(line)
    this.tryTail()
  }

  private line(line: string): void {
    if (!this.doc.length) {
      const spans = parseDocument(line)
      if (spans) return this.receiver.addSpans(spans)
    }
    this.doc.push(line)
    // A pretty-printed document ends with a closing bracket in column 0
    if (line[0] === '}' || line[0] === ']') {
      const spans = parseDocument(this.doc.join('\n'))
      if (spans) {
        this.doc = []
        this.receiver.addSpans(spans)
      }
    }
  }

  /** A file that doesn't end in a newline leaves its last line in the tail. */
  private tryTail(): void {
    if (!this.tail.trim()) return
    const spans = parseDocument([...this.doc, this.tail].join('\n'))
    if (spans) {
      this.doc = []
      this.tail = ''
      this.receiver.addSpans(spans)
    }
  }
}

function parseDocument(text: string): Span[] | null {
  const trimmed = text.trim()
  if (!trimmed || (trimmed[0] !== '{' && trimmed[0] !== '[')) return null
  try {
    JSON.parse(trimmed)
  } catch { return null }
  return spansFromText(trimmed)
}

/** Parse `AGENT_FLOW_OTEL_FILE`: paths joined by the platform delimiter. */
export function parseOtlpFilePaths(value: string | undefined): string[] {
  if (!value) return []
  return value.split(path.delimiter).map(p => p.trim()).filter(Boolean).map(p => path.resolve(p))
}

/** Read a whole OTLP file at once (for the import CLI). */
export function readOtlpFile(filePath: string): Span[] {
  return spansFromText(fs.readFileSync(filePath, 'utf8'))
}
