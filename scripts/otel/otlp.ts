/**
 * OTLP trace payloads → plain span records.
 *
 * Reads both encodings of an ExportTraceServiceRequest: the protobuf body
 * that OTLP/HTTP exporters send by default (decoded here without a protobuf
 * dependency) and the JSON mapping (OTLP/HTTP JSON, the collector's file
 * exporter, and most "export as OTLP JSON" buttons).
 */

export type AttrValue = string | number | boolean | null | AttrValue[] | { [key: string]: AttrValue }
export type Attributes = Record<string, AttrValue>

export interface SpanEvent {
  /** Nanoseconds since the Unix epoch */
  time: bigint
  name: string
  attributes: Attributes
}

export interface Span {
  traceId: string
  spanId: string
  /** Empty for a root span */
  parentSpanId: string
  name: string
  /** Nanoseconds since the Unix epoch */
  start: bigint
  end: bigint
  attributes: Attributes
  events: SpanEvent[]
  /** 0 unset, 1 ok, 2 error */
  statusCode: number
  statusMessage: string
  /** Resource attributes (service.name etc.) */
  resource: Attributes
  scope: string
}

// ─── JSON mapping ────────────────────────────────────────────────────────────

type Json = any // eslint-disable-line @typescript-eslint/no-explicit-any

function toBigInt(value: unknown): bigint {
  if (typeof value === 'bigint') return value
  if (typeof value === 'number' && Number.isFinite(value)) return BigInt(Math.trunc(value))
  if (typeof value === 'string' && /^\d+$/.test(value)) return BigInt(value)
  return BigInt(0)
}

/** Trace and span ids are hex in OTLP JSON, but protobuf's canonical JSON
 *  mapping (and some exporters) write bytes as base64. */
function idFromJson(value: unknown, bytes: number): string {
  if (typeof value !== 'string' || !value) return ''
  if (value.length === bytes * 2 && /^[0-9a-fA-F]+$/.test(value)) return value.toLowerCase()
  try {
    const hex = Buffer.from(value, 'base64').toString('hex')
    return hex.length === bytes * 2 ? hex : ''
  } catch { return '' }
}

function anyValueFromJson(v: Json): AttrValue {
  if (!v || typeof v !== 'object') return null
  if ('stringValue' in v) return String(v.stringValue)
  if ('boolValue' in v) return Boolean(v.boolValue)
  if ('intValue' in v) return Number(v.intValue)
  if ('doubleValue' in v) return Number(v.doubleValue)
  if ('arrayValue' in v) return ((v.arrayValue?.values ?? []) as Json[]).map(anyValueFromJson)
  if ('kvlistValue' in v) return attributesFromJson(v.kvlistValue?.values)
  if ('bytesValue' in v) return String(v.bytesValue)
  return null
}

function attributesFromJson(list: Json): Attributes {
  const out: Attributes = {}
  if (!Array.isArray(list)) return out
  for (const kv of list) {
    if (kv && typeof kv.key === 'string') out[kv.key] = anyValueFromJson(kv.value)
  }
  return out
}

function statusCodeFromJson(code: unknown): number {
  if (typeof code === 'number') return code
  if (code === 'STATUS_CODE_OK') return 1
  if (code === 'STATUS_CODE_ERROR') return 2
  return 0
}

/** Spans from one ExportTraceServiceRequest in the OTLP JSON encoding. */
export function spansFromJson(request: Json): Span[] {
  const spans: Span[] = []
  const resourceSpans = request?.resourceSpans ?? request?.resource_spans
  if (!Array.isArray(resourceSpans)) return spans
  for (const rs of resourceSpans) {
    const resource = attributesFromJson(rs?.resource?.attributes)
    for (const ss of rs?.scopeSpans ?? rs?.scope_spans ?? rs?.instrumentationLibrarySpans ?? []) {
      const scope = String(ss?.scope?.name ?? ss?.instrumentationLibrary?.name ?? '')
      for (const s of ss?.spans ?? []) {
        const traceId = idFromJson(s?.traceId ?? s?.trace_id, 16)
        const spanId = idFromJson(s?.spanId ?? s?.span_id, 8)
        if (!traceId || !spanId) continue
        spans.push({
          traceId,
          spanId,
          parentSpanId: idFromJson(s.parentSpanId ?? s.parent_span_id, 8),
          name: String(s.name ?? ''),
          start: toBigInt(s.startTimeUnixNano ?? s.start_time_unix_nano),
          end: toBigInt(s.endTimeUnixNano ?? s.end_time_unix_nano),
          attributes: attributesFromJson(s.attributes),
          events: (Array.isArray(s.events) ? s.events : []).map((e: Json) => ({
            time: toBigInt(e?.timeUnixNano ?? e?.time_unix_nano),
            name: String(e?.name ?? ''),
            attributes: attributesFromJson(e?.attributes),
          })),
          statusCode: statusCodeFromJson(s.status?.code),
          statusMessage: String(s.status?.message ?? ''),
          resource,
          scope,
        })
      }
    }
  }
  return spans
}

// ─── Protobuf ────────────────────────────────────────────────────────────────

/** A minimal protobuf wire-format reader, enough for the OTLP trace schema. */
class Reader {
  pos = 0
  constructor(readonly buf: Uint8Array, readonly end = buf.length) {}

  varint(): bigint {
    let result = BigInt(0)
    let shift = BigInt(0)
    for (let i = 0; i < 10; i++) {
      if (this.pos >= this.end) throw new Error('truncated varint')
      const byte = this.buf[this.pos++]
      result |= BigInt(byte & 0x7f) << shift
      if ((byte & 0x80) === 0) return result
      shift += BigInt(7)
    }
    throw new Error('varint too long')
  }

  uint(): number { return Number(this.varint()) }

  fixed64(): bigint {
    if (this.pos + 8 > this.end) throw new Error('truncated fixed64')
    const view = new DataView(this.buf.buffer, this.buf.byteOffset + this.pos, 8)
    this.pos += 8
    return view.getBigUint64(0, true)
  }

  double(): number {
    if (this.pos + 8 > this.end) throw new Error('truncated double')
    const view = new DataView(this.buf.buffer, this.buf.byteOffset + this.pos, 8)
    this.pos += 8
    return view.getFloat64(0, true)
  }

  bytes(): Uint8Array {
    const len = this.uint()
    if (this.pos + len > this.end) throw new Error('truncated length-delimited field')
    const out = this.buf.subarray(this.pos, this.pos + len)
    this.pos += len
    return out
  }

  string(): string { return Buffer.from(this.bytes()).toString('utf8') }

  sub(): Reader {
    const bytes = this.bytes()
    return new Reader(bytes)
  }

  skip(wireType: number): void {
    switch (wireType) {
      case 0: this.varint(); break
      case 1: this.pos += 8; break
      case 2: this.bytes(); break
      case 5: this.pos += 4; break
      default: throw new Error(`unsupported wire type ${wireType}`)
    }
    if (this.pos > this.end) throw new Error('truncated field')
  }

  /** Iterate fields: calls `fn(field, wireType)`, which must consume the value
   *  (or return false to have it skipped). */
  fields(fn: (field: number, wireType: number) => boolean | void): void {
    while (this.pos < this.end) {
      const tag = this.uint()
      const field = tag >>> 3
      const wireType = tag & 7
      if (fn(field, wireType) === false) this.skip(wireType)
    }
  }
}

function hex(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString('hex')
}

function readAnyValue(r: Reader): AttrValue {
  let value: AttrValue = null
  r.fields((field, wt) => {
    if (field === 1 && wt === 2) value = r.string()
    else if (field === 2 && wt === 0) value = r.varint() !== BigInt(0)
    else if (field === 3 && wt === 0) value = Number(BigInt.asIntN(64, r.varint()))
    else if (field === 4 && wt === 1) value = r.double()
    else if (field === 5 && wt === 2) {
      const arr: AttrValue[] = []
      const sub = r.sub()
      sub.fields((f, w) => {
        if (f === 1 && w === 2) { arr.push(readAnyValue(sub.sub())); return }
        return false
      })
      value = arr
    } else if (field === 6 && wt === 2) {
      const sub = r.sub()
      const obj: Attributes = {}
      sub.fields((f, w) => {
        if (f === 1 && w === 2) { const [k, v] = readKeyValue(sub.sub()); if (k) obj[k] = v; return }
        return false
      })
      value = obj
    } else if (field === 7 && wt === 2) value = Buffer.from(r.bytes()).toString('base64')
    else return false
  })
  return value
}

function readKeyValue(r: Reader): [string, AttrValue] {
  let key = ''
  let value: AttrValue = null
  r.fields((field, wt) => {
    if (field === 1 && wt === 2) key = r.string()
    else if (field === 2 && wt === 2) value = readAnyValue(r.sub())
    else return false
  })
  return [key, value]
}

function readAttributesInto(r: Reader, into: Attributes): void {
  const [k, v] = readKeyValue(r)
  if (k) into[k] = v
}

function readSpan(r: Reader, resource: Attributes, scope: string): Span {
  const span: Span = {
    traceId: '', spanId: '', parentSpanId: '', name: '',
    start: BigInt(0), end: BigInt(0), attributes: {}, events: [],
    statusCode: 0, statusMessage: '', resource, scope,
  }
  r.fields((field, wt) => {
    if (field === 1 && wt === 2) span.traceId = hex(r.bytes())
    else if (field === 2 && wt === 2) span.spanId = hex(r.bytes())
    else if (field === 4 && wt === 2) span.parentSpanId = hex(r.bytes())
    else if (field === 5 && wt === 2) span.name = r.string()
    else if (field === 7 && wt === 1) span.start = r.fixed64()
    else if (field === 8 && wt === 1) span.end = r.fixed64()
    else if (field === 9 && wt === 2) readAttributesInto(r.sub(), span.attributes)
    else if (field === 11 && wt === 2) {
      const e = r.sub()
      const event: SpanEvent = { time: BigInt(0), name: '', attributes: {} }
      e.fields((f, w) => {
        if (f === 1 && w === 1) event.time = e.fixed64()
        else if (f === 2 && w === 2) event.name = e.string()
        else if (f === 3 && w === 2) readAttributesInto(e.sub(), event.attributes)
        else return false
      })
      span.events.push(event)
    } else if (field === 15 && wt === 2) {
      const s = r.sub()
      s.fields((f, w) => {
        if (f === 2 && w === 2) span.statusMessage = s.string()
        else if (f === 3 && w === 0) span.statusCode = s.uint()
        else return false
      })
    } else return false
  })
  return span
}

/** Spans from one protobuf-encoded ExportTraceServiceRequest. Throws on a
 *  malformed body. */
export function spansFromProtobuf(body: Uint8Array): Span[] {
  const spans: Span[] = []
  const r = new Reader(body)
  r.fields((field, wt) => {
    if (field !== 1 || wt !== 2) return false
    const rs = r.sub()
    const resource: Attributes = {}
    const scopeReaders: Reader[] = []
    rs.fields((f, w) => {
      if (f === 1 && w === 2) {
        const res = rs.sub()
        res.fields((ff, ww) => {
          if (ff === 1 && ww === 2) { readAttributesInto(res.sub(), resource); return }
          return false
        })
      } else if (f === 2 && w === 2) scopeReaders.push(rs.sub())
      else return false
    })
    // Resource may follow the scope spans on the wire; read spans after it.
    for (const ss of scopeReaders) {
      let scope = ''
      const spanReaders: Reader[] = []
      ss.fields((f, w) => {
        if (f === 1 && w === 2) {
          const sc = ss.sub()
          sc.fields((ff, ww) => {
            if (ff === 1 && ww === 2) { scope = sc.string(); return }
            return false
          })
        } else if (f === 2 && w === 2) spanReaders.push(ss.sub())
        else return false
      })
      for (const sr of spanReaders) {
        const span = readSpan(sr, resource, scope)
        if (span.traceId && span.spanId) spans.push(span)
      }
    }
  })
  return spans
}

/**
 * Spans from a file or text body: one OTLP JSON request, or JSON Lines of
 * them (the collector's file exporter). Unparseable lines are skipped.
 */
export function spansFromText(text: string): Span[] {
  const trimmed = text.trim()
  if (!trimmed) return []
  try {
    return spansFromJsonValue(JSON.parse(trimmed))
  } catch { /* not one document; try JSON Lines */ }
  const spans: Span[] = []
  for (const line of trimmed.split(/\r?\n/)) {
    if (!line.trim()) continue
    try { spans.push(...spansFromJsonValue(JSON.parse(line))) } catch { /* skip */ }
  }
  return spans
}

function spansFromJsonValue(value: Json): Span[] {
  if (Array.isArray(value)) return value.flatMap(spansFromJsonValue)
  return spansFromJson(value)
}
