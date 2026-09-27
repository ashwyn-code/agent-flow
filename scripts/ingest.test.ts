import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import * as http from 'http'
import type { AddressInfo } from 'net'
import { IngestServer, MAX_BODY_BYTES, MAX_EVENTS_PER_REQUEST } from './ingest'
import type { AgentEvent } from '../extension/src/protocol'

interface Harness {
  url: string
  events: AgentEvent[]
  lifecycle: string[]
  ingest: IngestServer
  close: () => Promise<void>
}

async function harness(token?: string): Promise<Harness> {
  const events: AgentEvent[] = []
  const lifecycle: string[] = []
  const ingest = new IngestServer({
    onEvent: e => events.push(e),
    onLifecycle: (type, sessionId, label) => lifecycle.push(`${type}:${label}`),
  }, { token })
  const server = http.createServer((req, res) => ingest.handle(req, res))
  await new Promise<void>(r => server.listen(0, '127.0.0.1', () => r()))
  const { port } = server.address() as AddressInfo
  return { url: `http://127.0.0.1:${port}/ingest`, events, lifecycle, ingest, close: () => new Promise(r => server.close(() => r())) }
}

const RUN = [
  { time: 0, type: 'agent_spawn', payload: { name: 'triage', isMain: true } },
  { time: 0.1, type: 'message', payload: { agent: 'triage', role: 'user', content: 'Dashboards are slow' } },
  { time: 0.2, type: 'tool_call_start', payload: { agent: 'triage', tool: 'search' } },
  { time: 0.3, type: 'tool_call_end', payload: { agent: 'triage', tool: 'search', result: 'ok' } },
  { time: 0.4, type: 'agent_complete', payload: { name: 'triage' } },
]

function post(url: string, body: unknown, headers: Record<string, string> = {}) {
  return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body) })
}

test('loopback senders are accepted without a token and become sessions', async () => {
  const h = await harness()
  try {
    const res = await post(h.url, { session: { id: 'svc-a', label: 'checkout' }, events: RUN })
    assert.equal(res.status, 200)
    assert.deepEqual(await res.json(), { accepted: 5 })
    assert.deepEqual(h.events.map(e => e.type), RUN.map(e => e.type))
    assert.equal(new Set(h.events.map(e => e.sessionId)).size, 1)
    assert.match(h.events[0].sessionId!, /^ingest-[0-9a-f]{8}-1$/)
    assert.deepEqual(h.lifecycle, ['started:checkout', 'updated:Dashboards are slow', 'ended:Dashboards are slow'])
    assert.deepEqual(h.ingest.getSessions().map(s => [s.label, s.status]), [['Dashboards are slow', 'completed']])
  } finally { await h.close() }
})

test('each sender is its own session; batches from one sender continue it', async () => {
  const h = await harness()
  try {
    await post(h.url, { session: { id: 'svc-a' }, events: RUN.slice(0, 2) })
    await post(h.url, { session: { id: 'svc-b' }, events: RUN.slice(0, 1) })
    await post(h.url, { session: { id: 'svc-a' }, events: RUN.slice(2) })
    const bySession = new Map<string, string[]>()
    for (const e of h.events) bySession.set(e.sessionId!, [...(bySession.get(e.sessionId!) ?? []), e.type])
    assert.equal(bySession.size, 2)
    assert.deepEqual([...bySession.values()].map(v => v.length).sort(), [1, 5])
  } finally { await h.close() }
})

test('with a token, requests must present it', async () => {
  const h = await harness('s3cret')
  try {
    assert.equal((await post(h.url, { session: { id: 'x' }, events: RUN })).status, 401)
    assert.equal((await post(h.url, { session: { id: 'x' }, events: RUN }, { Authorization: 'Bearer nope' })).status, 401)
    assert.equal((await post(h.url, { session: { id: 'x' }, events: RUN }, { Authorization: 'Bearer s3cret' })).status, 200)
    assert.equal(h.events.length, 5)
  } finally { await h.close() }
})

test('other hosts are refused when no token is configured', () => {
  const ingest = new IngestServer({ onEvent: () => {}, onLifecycle: () => {} })
  let status = 0
  const req = { method: 'POST', headers: {}, socket: { remoteAddress: '10.0.0.5' }, on: () => {} } as unknown as http.IncomingMessage
  const res = { writeHead: (s: number) => { status = s }, end: () => {}, setHeader: () => {} } as unknown as http.ServerResponse
  ingest.handle(req, res)
  assert.equal(status, 403)
})

test('bad requests are rejected', async () => {
  const h = await harness()
  try {
    assert.equal((await fetch(h.url)).status, 405)
    assert.equal((await fetch(h.url, { method: 'POST', body: 'not json' })).status, 400)
    assert.equal((await post(h.url, { events: RUN })).status, 400)
    assert.equal((await post(h.url, { session: { id: 'x' }, events: 'nope' })).status, 400)
    const many = Array.from({ length: MAX_EVENTS_PER_REQUEST + 1 }, (_, i) => ({ time: i, type: 'message', payload: {} }))
    assert.equal((await post(h.url, { session: { id: 'x' }, events: many })).status, 413)
    const huge = 'x'.repeat(MAX_BODY_BYTES + 10)
    const res = await fetch(h.url, { method: 'POST', body: JSON.stringify({ session: { id: 'x' }, events: [], pad: huge }) })
    assert.equal(res.status, 413)
    assert.equal(h.events.length, 0)
  } finally { await h.close() }
})

test('malformed events in a batch are skipped', async () => {
  const h = await harness()
  try {
    const res = await post(h.url, { session: { id: 'x' }, events: [RUN[0], { type: 'message' }, 'junk', RUN[4]] })
    assert.deepEqual(await res.json(), { accepted: 2 })
    assert.deepEqual(h.events.map(e => e.type), ['agent_spawn', 'agent_complete'])
  } finally { await h.close() }
})
