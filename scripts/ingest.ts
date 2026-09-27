/**
 * `POST /ingest`: receives batches of Agent Flow events over HTTP (the
 * adapters' `url=` / `AGENT_FLOW_URL` transport), so agents running in other
 * processes, containers or hosts can stream into one relay.
 *
 * Request body:
 *   { "session": { "id": "<sender id>", "label"?: "<tab label>" },
 *     "events": [ { "time": 1.2, "type": "tool_call_start", "payload": { ... } }, ... ] }
 *
 * Access:
 *   - With a token (AGENT_FLOW_INGEST_TOKEN / --ingest-token), requests must
 *     send `Authorization: Bearer <token>`.
 *   - Without one, only loopback clients are accepted.
 *
 * Each sender id becomes its own session tab, tracked like an event log.
 */
import * as crypto from 'crypto'
import type * as http from 'http'

import type { AgentEvent, SessionInfo } from '../extension/src/protocol'
import { SessionTracker, type EventLogCallbacks } from './event-log-watcher'

export const MAX_BODY_BYTES = 5 * 1024 * 1024
export const MAX_EVENTS_PER_REQUEST = 5000
const MAX_SESSIONS = 500
const LABEL_MAX = 60

export interface IngestOptions {
  token?: string
}

function isLoopback(address: string | undefined): boolean {
  if (!address) return false
  return address === '127.0.0.1' || address === '::1' || address === '::ffff:127.0.0.1' || address.startsWith('127.')
}

function tokenMatches(header: string | undefined, token: string): boolean {
  const match = /^Bearer\s+(.+)$/i.exec(header ?? '')
  if (!match) return false
  const given = Buffer.from(match[1].trim())
  const expected = Buffer.from(token)
  return given.length === expected.length && crypto.timingSafeEqual(given, expected)
}

export function send(res: http.ServerResponse, status: number, body: Record<string, unknown>): void {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(body))
}

function toEvent(raw: unknown): AgentEvent | null {
  if (!raw || typeof raw !== 'object') return null
  const { time, type, payload } = raw as Record<string, unknown>
  if (typeof time !== 'number' || typeof type !== 'string') return null
  return { time, type: type as AgentEvent['type'], payload: payload && typeof payload === 'object' ? payload as Record<string, unknown> : {} }
}

/**
 * Check a request against the ingest access rules (bearer token when one is
 * configured, loopback clients only otherwise). Sends the refusal and returns
 * false when it isn't allowed. Shared by `/ingest` and OTLP `/v1/traces`.
 */
export function authorize(req: http.IncomingMessage, res: http.ServerResponse, token: string | undefined): boolean {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST')
    send(res, 405, { error: 'use POST' })
    return false
  }
  if (token) {
    if (!tokenMatches(req.headers.authorization, token)) {
      send(res, 401, { error: 'missing or wrong bearer token' })
      return false
    }
  } else if (!isLoopback(req.socket.remoteAddress)) {
    send(res, 403, { error: 'set an ingest token to accept events from other hosts' })
    return false
  }
  return true
}

/** Read a request body of at most `maxBytes`, answering 413 past that. */
export function readBody(req: http.IncomingMessage, res: http.ServerResponse, maxBytes: number, onBody: (body: Buffer) => void): void {
  const declared = Number(req.headers['content-length'] ?? 0)
  if (declared > maxBytes) return send(res, 413, { error: `body over ${maxBytes} bytes` })
  const chunks: Buffer[] = []
  let size = 0
  let aborted = false
  req.on('data', (chunk: Buffer) => {
    size += chunk.length
    if (size > maxBytes && !aborted) {
      aborted = true
      send(res, 413, { error: `body over ${maxBytes} bytes` })
      req.destroy()
    } else if (!aborted) {
      chunks.push(chunk)
    }
  })
  req.on('end', () => {
    if (!aborted) onBody(Buffer.concat(chunks))
  })
}

export class IngestServer {
  private readonly trackers = new Map<string, SessionTracker>()

  constructor(private readonly callbacks: EventLogCallbacks, private readonly options: IngestOptions = {}) {}

  getSessions(): SessionInfo[] {
    return [...this.trackers.values()].flatMap(t => t.getSessions())
  }

  /** Accept (or refuse) one request. */
  handle(req: http.IncomingMessage, res: http.ServerResponse): void {
    if (!authorize(req, res, this.options.token)) return
    readBody(req, res, MAX_BODY_BYTES, raw => {
      let body: Record<string, unknown>
      try {
        body = JSON.parse(raw.toString('utf8'))
      } catch {
        return send(res, 400, { error: 'body is not JSON' })
      }
      const session = body?.session as Record<string, unknown> | undefined
      const events = body?.events
      if (!session || typeof session.id !== 'string' || !session.id || !Array.isArray(events)) {
        return send(res, 400, { error: 'expected { session: { id }, events: [...] }' })
      }
      if (events.length > MAX_EVENTS_PER_REQUEST) {
        return send(res, 413, { error: `more than ${MAX_EVENTS_PER_REQUEST} events in one request` })
      }
      const tracker = this.tracker(session.id, typeof session.label === 'string' ? session.label : '')
      if (!tracker) return send(res, 429, { error: 'too many sessions' })
      let accepted = 0
      for (const item of events) {
        const event = toEvent(item)
        if (event) {
          tracker.handle(event)
          accepted++
        }
      }
      send(res, 200, { accepted })
    })
  }

  private tracker(senderId: string, label: string): SessionTracker | null {
    let tracker = this.trackers.get(senderId)
    if (!tracker) {
      if (this.trackers.size >= MAX_SESSIONS) return null
      const hash = crypto.createHash('sha256').update(senderId).digest('hex').slice(0, 8)
      const tab = (label || 'remote session').slice(0, LABEL_MAX)
      tracker = new SessionTracker(`ingest-${hash}`, tab, this.callbacks)
      this.trackers.set(senderId, tracker)
    }
    return tracker
  }
}
