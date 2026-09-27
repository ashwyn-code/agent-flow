import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import * as fs from 'fs'
import * as http from 'http'
import * as os from 'os'
import * as path from 'path'
import * as zlib from 'zlib'
import type { AddressInfo } from 'net'

import type { AgentEvent } from '../../extension/src/protocol'
import { OtlpFileWatcher, OtlpReceiver, type OtlpReceiverOptions } from './receiver'

const FIXTURES = path.join(__dirname, 'fixtures')
const fixture = (name: string) => zlib.gunzipSync(fs.readFileSync(path.join(FIXTURES, name)))
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

async function until(check: () => boolean, ms = 2000): Promise<void> {
  const deadline = Date.now() + ms
  while (!check()) {
    if (Date.now() > deadline) throw new Error('timed out waiting')
    await sleep(10)
  }
}

interface Harness {
  url: string
  events: AgentEvent[]
  lifecycle: string[]
  receiver: OtlpReceiver
  close: () => Promise<void>
}

async function harness(options: OtlpReceiverOptions = {}): Promise<Harness> {
  const events: AgentEvent[] = []
  const lifecycle: string[] = []
  const receiver = new OtlpReceiver({
    onEvent: e => events.push(e),
    onLifecycle: (type, _id, label) => lifecycle.push(`${type}:${label}`),
  }, { settleMs: 0, speed: 0, ...options })
  const server = http.createServer((req, res) => receiver.handle(req, res))
  await new Promise<void>(r => server.listen(0, '127.0.0.1', () => r()))
  const { port } = server.address() as AddressInfo
  return {
    url: `http://127.0.0.1:${port}/v1/traces`, events, lifecycle, receiver,
    close: () => { receiver.dispose(); return new Promise(r => server.close(() => r())) },
  }
}

/** A small trace in OTLP JSON: an agent with one tool call. */
function trace(traceId: string, opts: { root?: boolean; start?: number } = {}) {
  const t = (s: number) => String(BigInt('1700000000000000000') + BigInt(Math.round(((opts.start ?? 0) + s) * 1e9)))
  const attr = (key: string, value: string) => ({ key, value: { stringValue: value } })
  const spans = [
    { traceId, spanId: '00000000000000a2', parentSpanId: '00000000000000a1', name: 'execute_tool search', startTimeUnixNano: t(1), endTimeUnixNano: t(2),
      attributes: [attr('gen_ai.operation.name', 'execute_tool'), attr('gen_ai.tool.name', 'search'), attr('gen_ai.tool.call.result', 'found')] },
  ]
  if (opts.root !== false) {
    spans.push({ traceId, spanId: '00000000000000a1', parentSpanId: '', name: 'invoke_agent helper', startTimeUnixNano: t(0), endTimeUnixNano: t(10),
      attributes: [attr('gen_ai.operation.name', 'invoke_agent'), attr('gen_ai.agent.name', 'helper'),
        attr('gen_ai.input.messages', JSON.stringify([{ role: 'user', parts: [{ type: 'text', content: 'Find the thing' }] }]))] })
  }
  return { resourceSpans: [{ resource: { attributes: [] }, scopeSpans: [{ scope: { name: 't' }, spans }] }] }
}

const TRACE_A = 'a'.repeat(32)

test('protobuf over HTTP (gzip) becomes a session', async () => {
  const h = await harness()
  try {
    const res = await fetch(h.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-protobuf', 'Content-Encoding': 'gzip' },
      body: zlib.gzipSync(fixture('google-adk.otlp.pb.gz')),
    })
    assert.equal(res.status, 200)
    assert.equal(res.headers.get('content-type'), 'application/x-protobuf')
    assert.equal((await res.arrayBuffer()).byteLength, 0)
    await until(() => h.lifecycle.some(l => l.startsWith('ended')))
    assert.equal(h.lifecycle[0], 'started:trip_planner')
    assert.match(h.lifecycle[1], /^updated:Plan a 4-day trip to Lisbon/)
    assert.equal(h.events[0].type, 'agent_spawn')
    assert.equal(new Set(h.events.map(e => e.sessionId)).size, 1)
    assert.match(h.events[0].sessionId!, /^otel-[0-9a-f]{8}-1$/)
    assert.deepEqual(h.receiver.getSessions().map(s => s.status), ['completed'])
  } finally { await h.close() }
})

test('OTLP JSON over HTTP, and each trace is its own session', async () => {
  const h = await harness()
  try {
    for (const id of [TRACE_A, 'b'.repeat(32)]) {
      const res = await fetch(h.url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(trace(id)) })
      assert.equal(res.status, 200)
      assert.deepEqual(await res.json(), {})
    }
    await until(() => h.receiver.getSessions().length === 2)
    assert.equal(new Set(h.events.map(e => e.sessionId)).size, 2)
    assert.deepEqual(h.events.filter(e => e.type === 'tool_call_end').map(e => e.payload.result), ['found', 'found'])
  } finally { await h.close() }
})

test('a trace waits for its root span, however the spans are batched', async () => {
  const h = await harness({ idleMs: 10_000 })
  try {
    h.receiver.addSpans([]) // no-op
    const { spansFromJson } = await import('./otlp')
    h.receiver.addSpans(spansFromJson(trace(TRACE_A, { root: false })))
    await sleep(30)
    assert.equal(h.events.length, 0, 'child spans alone are held')
    h.receiver.addSpans(spansFromJson(trace(TRACE_A)).filter(s => !s.parentSpanId))
    await until(() => h.events.length > 0)
    const types = h.events.map(e => e.type)
    assert.deepEqual(types, ['agent_spawn', 'message', 'tool_call_start', 'tool_call_end', 'agent_complete'])
    // Spans for a trace already shown are counted, not re-shown
    h.receiver.addSpans(spansFromJson(trace(TRACE_A)))
    await sleep(20)
    assert.equal(h.receiver.lateSpans, 2)
    assert.equal(h.receiver.getSessions().length, 1)
  } finally { await h.close() }
})

test('a trace whose root never arrives is shown once it goes quiet', async () => {
  const h = await harness({ idleMs: 50 })
  try {
    const { spansFromJson } = await import('./otlp')
    h.receiver.addSpans(spansFromJson(trace(TRACE_A, { root: false })))
    await sleep(20)
    assert.equal(h.events.length, 0)
    await until(() => h.events.length > 0)
    assert.ok(h.events.some(e => e.type === 'tool_call_start'))
  } finally { await h.close() }
})

test('replay keeps the order and pace, with long pauses shortened', async () => {
  const h = await harness({ speed: 1, maxGapS: 0.05 })
  try {
    const { spansFromJson } = await import('./otlp')
    const started = Date.now()
    h.receiver.addSpans(spansFromJson(trace(TRACE_A)))   // 10 s of trace time
    await until(() => h.events.some(e => e.type === 'agent_complete'))
    const took = Date.now() - started
    assert.ok(took >= 100 && took < 1500, `took ${took} ms`)
    assert.deepEqual(h.events.map(e => e.type), ['agent_spawn', 'message', 'tool_call_start', 'tool_call_end', 'agent_complete'])
  } finally { await h.close() }
})

test('a declared workflow from an earlier request is drawn on the run', async () => {
  const h = await harness()
  try {
    const text = fixture('agent-framework.otlp.jsonl.gz').toString('utf8')
    const lines = text.trim().split('\n')
    // The collector may batch the build spans and the run separately
    const isBuild = (l: string) => l.includes('"workflow.build"')
    for (const group of [lines.filter(isBuild), lines.filter(l => !isBuild(l))]) {
      for (const line of group) {
        await fetch(h.url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: line })
      }
    }
    await until(() => h.lifecycle.some(l => l.startsWith('ended')))
    const structure = h.events.find(e => e.type === 'graph_structure' && e.payload.agent === 'incident_response')!
    const edges = (structure.payload.edges as { source: string; target: string; label?: string }[])
    assert.ok(edges.some(e => e.source === 'triage' && e.target === 'auto_ack' && e.label === 'default'))
    assert.equal(h.receiver.getSessions().length, 1, 'build-only traces are not sessions')
  } finally { await h.close() }
})

test('access and bad requests', async () => {
  const h = await harness({ token: 's3cret' })
  try {
    const body = JSON.stringify(trace(TRACE_A))
    const json = { 'Content-Type': 'application/json' }
    assert.equal((await fetch(h.url)).status, 405)
    assert.equal((await fetch(h.url, { method: 'POST', headers: json, body })).status, 401)
    assert.equal((await fetch(h.url, { method: 'POST', headers: { ...json, Authorization: 'Bearer nope' }, body })).status, 401)
    const auth = { Authorization: 'Bearer s3cret' }
    assert.equal((await fetch(h.url, { method: 'POST', headers: { ...json, ...auth }, body: '{nope' })).status, 400)
    assert.equal((await fetch(h.url, { method: 'POST', headers: { 'Content-Type': 'application/x-protobuf', ...auth }, body: Buffer.from([0x0a, 0xff]) })).status, 400)
    assert.equal((await fetch(h.url, { method: 'POST', headers: { ...json, ...auth, 'Content-Encoding': 'gzip' }, body })).status, 400)
    assert.equal((await fetch(h.url, { method: 'POST', headers: { ...json, ...auth }, body })).status, 200)
    await until(() => h.events.length > 0)
  } finally { await h.close() }
})

test('other hosts are refused without a token', () => {
  const receiver = new OtlpReceiver({ onEvent: () => {}, onLifecycle: () => {} })
  let status = 0
  const req = { method: 'POST', headers: {}, socket: { remoteAddress: '10.0.0.5' }, on: () => {} } as unknown as http.IncomingMessage
  const res = { writeHead: (s: number) => { status = s }, end: () => {}, setHeader: () => {} } as unknown as http.ServerResponse
  receiver.handle(req, res)
  assert.equal(status, 403)
})

test('file watcher: JSON Lines appended over time, and a pretty-printed file', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-flow-otel-'))
  const events: AgentEvent[] = []
  const receiver = new OtlpReceiver({ onEvent: e => events.push(e), onLifecycle: () => {} }, { settleMs: 0, speed: 0 })
  try {
    const jsonl = path.join(dir, 'traces.jsonl')
    const watcher = new OtlpFileWatcher(jsonl, receiver)
    watcher.poll()   // file doesn't exist yet
    fs.writeFileSync(jsonl, JSON.stringify(trace(TRACE_A, { root: false })) + '\n')
    watcher.poll()
    fs.appendFileSync(jsonl, JSON.stringify(trace(TRACE_A)) + '\n' + JSON.stringify(trace('b'.repeat(32))))  // no trailing newline
    watcher.poll()
    await until(() => new Set(events.map(e => e.sessionId)).size === 2)

    const pretty = path.join(dir, 'export.json')
    fs.writeFileSync(pretty, JSON.stringify(trace('c'.repeat(32)), null, 2))
    const before = events.length
    new OtlpFileWatcher(pretty, receiver).poll()
    await until(() => events.length > before)
    assert.equal(new Set(events.map(e => e.sessionId)).size, 3)
  } finally {
    receiver.dispose()
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
