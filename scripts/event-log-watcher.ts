/**
 * Tails a generic Agent Flow JSONL event log (one AgentEvent per line, as
 * written by adapters such as adapters/langgraph) and exposes it to the relay
 * as a session, so logs from any framework can be viewed without VS Code.
 *
 * Each run in the file is its own session: truncating the file (e.g. the
 * LangGraph adapter's `truncate=True`) ends the current session and starts a
 * fresh one, so the view resets instead of piling new agents onto old state.
 */
import * as crypto from 'crypto'
import * as fs from 'fs'
import * as path from 'path'

import { AgentEvent, SessionInfo } from '../extension/src/protocol'
import { readNewFileLines } from '../extension/src/fs-utils'

const DEFAULT_POLL_MS = 250
const LABEL_MAX = 60
/** Bytes from the start of the file used to spot a rewrite that grew past the
 *  old size between polls (a size check alone would miss it). Early events
 *  carry per-run millisecond timestamps, so this differs between runs. */
const HEAD_BYTES = 1024

export type EventLogLifecycle = 'started' | 'ended' | 'updated'

export interface EventLogCallbacks {
  onEvent: (event: AgentEvent) => void
  onLifecycle: (type: EventLogLifecycle, sessionId: string, label: string) => void
}

interface RunState {
  id: string
  label: string
  labelSet: boolean
  mainAgent: string | null
  status: SessionInfo['status']
  startTime: number
  lastActivityTime: number
}

/** Parse the `AGENT_FLOW_EVENT_LOG` value: one or more paths joined by the
 *  platform path delimiter (`:` on macOS/Linux, `;` on Windows). */
export function parseEventLogPaths(value: string | undefined): string[] {
  if (!value) return []
  return value.split(path.delimiter).map(p => p.trim()).filter(Boolean).map(p => path.resolve(p))
}

function parseLine(line: string): AgentEvent | null {
  try {
    const parsed = JSON.parse(line.trim())
    if (parsed && typeof parsed.type === 'string' && typeof parsed.time === 'number') {
      return { time: parsed.time, type: parsed.type, payload: parsed.payload ?? {} }
    }
  } catch { /* skip malformed lines */ }
  return null
}

export class EventLogWatcher {
  private readonly idPrefix: string
  private fileSize = 0
  private tail = ''
  private runCount = 0
  private run: RunState | null = null
  private head: Buffer | null = null
  private pollTimer: NodeJS.Timeout | null = null
  private watcher: fs.FSWatcher | null = null

  constructor(
    readonly filePath: string,
    private readonly callbacks: EventLogCallbacks,
    private readonly pollMs = DEFAULT_POLL_MS,
  ) {
    const hash = crypto.createHash('sha256').update(path.resolve(filePath)).digest('hex').slice(0, 8)
    this.idPrefix = `eventlog-${hash}`
  }

  /** Read what's already in the file, then keep following it. The file may
   *  not exist yet — it's picked up once something starts writing it. */
  start(): void {
    this.poll()
    this.pollTimer = setInterval(() => this.poll(), this.pollMs)
  }

  private watchFile(): void {
    if (this.watcher) return
    try {
      this.watcher = fs.watch(this.filePath, () => this.poll())
      this.watcher.on('error', () => { this.watcher?.close(); this.watcher = null })
    } catch { /* polling still covers it */ }
  }

  getSessions(): SessionInfo[] {
    if (!this.run) return []
    const { id, label, status, startTime, lastActivityTime } = this.run
    return [{ id, label, status, startTime, lastActivityTime }]
  }

  dispose(): void {
    if (this.pollTimer) clearInterval(this.pollTimer)
    this.pollTimer = null
    this.watcher?.close()
    this.watcher = null
  }

  /** Exposed for tests; normally driven by the poll timer. */
  poll(): void {
    if (this.fileSize > 0 && this.head && !this.head.equals(this.readHead(this.head.length))) {
      this.resetForNewRun()
    }
    const result = readNewFileLines(this.filePath, this.fileSize, this.tail)
    if (!result) return
    this.watchFile()
    if (result.newSize === 0 && this.fileSize > 0) {
      this.resetForNewRun()
      return
    }
    this.fileSize = result.newSize
    this.tail = result.tail
    if (!this.head || this.head.length < HEAD_BYTES) this.head = this.readHead(HEAD_BYTES)
    for (const line of result.lines) {
      const event = parseLine(line)
      if (event) this.handleEvent(event)
    }
  }

  /** The file was truncated or rewritten: the writer started a new run. */
  private resetForNewRun(): void {
    this.endRun()
    this.run = null
    this.fileSize = 0
    this.tail = ''
    this.head = null
  }

  private readHead(length: number): Buffer {
    try {
      const fd = fs.openSync(this.filePath, 'r')
      try {
        const buf = Buffer.alloc(length)
        const n = fs.readSync(fd, buf, 0, length, 0)
        return buf.subarray(0, n)
      } finally { fs.closeSync(fd) }
    } catch { return Buffer.alloc(0) }
  }

  private beginRun(): RunState {
    this.runCount++
    const now = Date.now()
    const run: RunState = {
      id: `${this.idPrefix}-${this.runCount}`,
      label: path.basename(this.filePath),
      labelSet: false,
      mainAgent: null,
      status: 'active',
      startTime: now,
      lastActivityTime: now,
    }
    this.run = run
    this.callbacks.onLifecycle('started', run.id, run.label)
    return run
  }

  private endRun(): void {
    const run = this.run
    if (run && run.status === 'active') {
      run.status = 'completed'
      this.callbacks.onLifecycle('ended', run.id, run.label)
    }
  }

  private handleEvent(event: AgentEvent): void {
    const run = this.run ?? this.beginRun()
    run.lastActivityTime = Date.now()
    const payload = event.payload as Record<string, unknown>

    if (event.type === 'agent_spawn' && payload.isMain === true && typeof payload.name === 'string') {
      run.mainAgent = payload.name
      if (run.status === 'completed') {
        // Same session invoked again (e.g. a reused callback handler).
        run.status = 'active'
        this.callbacks.onLifecycle('started', run.id, run.label)
      }
    }

    if (!run.labelSet && event.type === 'message' && payload.role === 'user' && typeof payload.content === 'string') {
      const firstLine = payload.content.trim().split(/\r?\n/)[0] ?? ''
      if (firstLine) {
        run.label = firstLine.length > LABEL_MAX ? firstLine.slice(0, LABEL_MAX - 1) + '…' : firstLine
        run.labelSet = true
        this.callbacks.onLifecycle('updated', run.id, run.label)
      }
    }

    this.callbacks.onEvent({ ...event, sessionId: run.id })

    if (event.type === 'agent_complete' && run.mainAgent !== null && payload.name === run.mainAgent) {
      this.endRun()
    }
  }
}
