import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import * as fs from 'fs'
import * as path from 'path'
import * as zlib from 'zlib'

import { spansFromJson, spansFromProtobuf, spansFromText, type Span } from './otlp'

const FIXTURES = path.join(__dirname, 'fixtures')

function fixture(name: string): Buffer {
  return zlib.gunzipSync(fs.readFileSync(path.join(FIXTURES, name)))
}

const byId = (a: Span, b: Span) => a.spanId.localeCompare(b.spanId)

test('protobuf and JSON encodings of the same export decode identically', () => {
  const fromJson = spansFromText(fixture('google-adk.otlp.jsonl.gz').toString('utf8')).sort(byId)
  const fromProto = spansFromProtobuf(fixture('google-adk.otlp.pb.gz')).sort(byId)
  assert.equal(fromProto.length, 82)
  assert.deepEqual(fromProto, fromJson)
  const tool = fromProto.find(s => s.name === 'execute_tool lookup_profile')!
  assert.equal(tool.attributes['gen_ai.tool.name'], 'lookup_profile')
  assert.equal(tool.resource['service.name'], 'google-adk-demo')
  assert.match(tool.traceId, /^[0-9a-f]{32}$/)
  assert.match(tool.parentSpanId, /^[0-9a-f]{16}$/)
  assert.ok(tool.end > tool.start)
})

test('OTLP JSON with hex ids and every attribute type', () => {
  const [span] = spansFromJson({
    resourceSpans: [{
      resource: { attributes: [{ key: 'service.name', value: { stringValue: 'svc' } }] },
      scopeSpans: [{
        scope: { name: 'my.scope' },
        spans: [{
          traceId: '5B8EFFF798038103D269B633813FC60C',
          spanId: 'EEE19B7EC3C1B174',
          name: 'invoke_agent triage',
          startTimeUnixNano: '1700000000000000000',
          endTimeUnixNano: '1700000001500000000',
          attributes: [
            { key: 's', value: { stringValue: 'x' } },
            { key: 'i', value: { intValue: '-42' } },
            { key: 'd', value: { doubleValue: 1.5 } },
            { key: 'b', value: { boolValue: true } },
            { key: 'a', value: { arrayValue: { values: [{ stringValue: 'p' }, { intValue: 2 }] } } },
            { key: 'kv', value: { kvlistValue: { values: [{ key: 'k', value: { stringValue: 'v' } }] } } },
          ],
          events: [{ timeUnixNano: '1700000000500000000', name: 'gen_ai.user.message', attributes: [{ key: 'content', value: { stringValue: 'hi' } }] }],
          status: { code: 2, message: 'boom' },
        }],
      }],
    }],
  })
  assert.equal(span.traceId, '5b8efff798038103d269b633813fc60c')
  assert.equal(span.spanId, 'eee19b7ec3c1b174')
  assert.equal(span.parentSpanId, '')
  assert.equal(span.end - span.start, BigInt(1_500_000_000))
  assert.deepEqual(span.attributes, { s: 'x', i: -42, d: 1.5, b: true, a: ['p', 2], kv: { k: 'v' } })
  assert.equal(span.events[0].attributes.content, 'hi')
  assert.equal(span.statusCode, 2)
  assert.equal(span.statusMessage, 'boom')
  assert.equal(span.scope, 'my.scope')
  assert.equal(span.resource['service.name'], 'svc')
})

test('status codes may be names, and spans without ids are skipped', () => {
  const spans = spansFromJson({
    resourceSpans: [{ scopeSpans: [{ spans: [
      { traceId: 'AAAAAAAAAAAAAAAAAAAAAQ==', spanId: 'AAAAAAAAAAE=', name: 'ok', status: { code: 'STATUS_CODE_OK' } },
      { name: 'no ids' },
    ] }] }],
  })
  assert.equal(spans.length, 1)
  assert.equal(spans[0].traceId, '00000000000000000000000000000001')
  assert.equal(spans[0].statusCode, 1)
})

test('text input: one pretty-printed document, or JSON Lines', () => {
  const request = (name: string) => ({
    resourceSpans: [{ scopeSpans: [{ spans: [{ traceId: 'ab'.repeat(16), spanId: 'cd'.repeat(8), name }] }] }],
  })
  assert.deepEqual(spansFromText(JSON.stringify(request('pretty'), null, 2)).map(s => s.name), ['pretty'])
  const lines = [request('one'), request('two')].map(r => JSON.stringify(r)).join('\n') + '\nnot json\n'
  assert.deepEqual(spansFromText(lines).map(s => s.name), ['one', 'two'])
})

// ─── A tiny protobuf writer, to cover wire types the fixture doesn't ────────

function varint(n: bigint): number[] {
  const out: number[] = []
  let v = BigInt.asUintN(64, n)
  do {
    let byte = Number(v & BigInt(0x7f))
    v >>= BigInt(7)
    if (v > BigInt(0)) byte |= 0x80
    out.push(byte)
  } while (v > BigInt(0))
  return out
}
const tag = (field: number, wire: number) => varint(BigInt((field << 3) | wire))
const len = (field: number, bytes: number[]) => [...tag(field, 2), ...varint(BigInt(bytes.length)), ...bytes]
const str = (field: number, s: string) => len(field, [...Buffer.from(s, 'utf8')])
const fixed64 = (field: number, n: bigint) => {
  const b = Buffer.alloc(8)
  b.writeBigUInt64LE(n)
  return [...tag(field, 1), ...b]
}
const double = (field: number, n: number) => {
  const b = Buffer.alloc(8)
  b.writeDoubleLE(n)
  return [...tag(field, 1), ...b]
}
const kv = (key: string, value: number[]) => len(9, [...str(1, key), ...len(2, value)])

test('protobuf: all AnyValue kinds, unknown fields skipped', () => {
  const span = [
    ...len(1, Array(16).fill(1)), ...len(2, Array(8).fill(2)), ...len(4, Array(8).fill(3)),
    ...str(5, 'execute_tool lookup'),
    ...tag(6, 0), ...varint(BigInt(1)), // kind (ignored)
    ...fixed64(7, BigInt('1700000000000000000')), ...fixed64(8, BigInt('1700000000250000000')),
    ...kv('neg', [...tag(3, 0), ...varint(BigInt(-7))]),
    ...kv('flag', [...tag(2, 0), ...varint(BigInt(1))]),
    ...kv('ratio', double(4, 0.25)),
    ...kv('list', len(5, [...len(1, str(1, 'a')), ...len(1, [...tag(3, 0), ...varint(BigInt(3))])])),
    ...kv('map', len(6, len(1, [...str(1, 'inner'), ...len(2, str(1, 'v'))]))),
    ...tag(99, 0), ...varint(BigInt(12345)), // unknown field
    ...len(11, [...fixed64(1, BigInt('1700000000100000000')), ...str(2, 'exception'),
      ...len(3, [...str(1, 'exception.message'), ...len(2, str(1, 'bad'))])]),
    ...len(15, [...str(2, 'failed'), ...tag(3, 0), ...varint(BigInt(2))]),
  ]
  const request = len(1, [
    ...len(1, len(1, [...str(1, 'service.name'), ...len(2, str(1, 'svc'))])),
    ...len(2, [...len(1, str(1, 'scope')), ...len(2, span)]),
  ])
  const [s] = spansFromProtobuf(Uint8Array.from(request))
  assert.equal(s.traceId, '01'.repeat(16))
  assert.equal(s.parentSpanId, '03'.repeat(8))
  assert.equal(s.name, 'execute_tool lookup')
  assert.equal(s.end - s.start, BigInt(250_000_000))
  assert.deepEqual(s.attributes, { neg: -7, flag: true, ratio: 0.25, list: ['a', 3], map: { inner: 'v' } })
  assert.equal(s.events[0].name, 'exception')
  assert.equal(s.events[0].attributes['exception.message'], 'bad')
  assert.equal(s.statusCode, 2)
  assert.equal(s.statusMessage, 'failed')
  assert.equal(s.resource['service.name'], 'svc')
  assert.equal(s.scope, 'scope')
})

test('protobuf: a truncated body throws', () => {
  const body = fixture('google-adk.otlp.pb.gz')
  assert.throws(() => spansFromProtobuf(body.subarray(0, body.length - 3)))
})
