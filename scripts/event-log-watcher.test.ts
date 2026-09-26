import { test, beforeEach, afterEach } from 'node:test'
import { strict as assert } from 'node:assert'
import * as fs from 'fs'
import * as os from 'os'
import * as path from 'path'
import { EventLogWatcher, parseEventLogPaths } from './event-log-watcher'
import type { AgentEvent } from '../extension/src/protocol'

let dir: string
let file: string
let events: AgentEvent[]
let lifecycle: string[]
let watcher: EventLogWatcher

const line = (time: number, type: string, payload: Record<string, unknown>) =>
  JSON.stringify({ time, type, payload }) + '\n'

const RUN = [
  line(0, 'agent_spawn', { name: 'supervisor', isMain: true, task: 'Write a report' }),
  line(0.001, 'message', { agent: 'supervisor', role: 'user', content: 'Write a report\non churn' }),
  line(0.01, 'tool_call_start', { agent: 'supervisor', tool: 'search', args: 'q' }),
  line(0.02, 'tool_call_end', { agent: 'supervisor', tool: 'search', result: 'ok' }),
  line(0.03, 'agent_complete', { name: 'supervisor' }),
]

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-flow-eventlog-'))
  file = path.join(dir, 'events.jsonl')
  events = []
  lifecycle = []
  watcher = new EventLogWatcher(file, {
    onEvent: e => events.push(e),
    onLifecycle: (type, sessionId, label) => lifecycle.push(`${type}:${sessionId.split('-').pop()}:${label}`),
  }, 60_000) // timer effectively off; tests drive poll() directly
})

afterEach(() => {
  watcher.dispose()
  fs.rmSync(dir, { recursive: true, force: true })
})

test('replays existing content as one session with a label from the user prompt', () => {
  fs.writeFileSync(file, RUN.join(''))
  watcher.start()

  assert.deepEqual(events.map(e => e.type), ['agent_spawn', 'message', 'tool_call_start', 'tool_call_end', 'agent_complete'])
  const ids = new Set(events.map(e => e.sessionId))
  assert.equal(ids.size, 1)
  assert.match([...ids][0]!, /^eventlog-[0-9a-f]{8}-1$/)
  assert.deepEqual(events[0].payload, { name: 'supervisor', isMain: true, task: 'Write a report' })
  assert.equal(events[2].time, 0.01)
  assert.deepEqual(lifecycle, ['started:1:events.jsonl', 'updated:1:Write a report', 'ended:1:Write a report'])
  assert.deepEqual(watcher.getSessions().map(s => [s.label, s.status]), [['Write a report', 'completed']])
})

test('waits for a missing file, then follows appends', () => {
  watcher.start()
  assert.equal(events.length, 0)
  assert.deepEqual(watcher.getSessions(), [])

  fs.writeFileSync(file, RUN.slice(0, 2).join(''))
  watcher.poll()
  assert.equal(events.length, 2)
  assert.equal(watcher.getSessions()[0].status, 'active')

  fs.appendFileSync(file, RUN.slice(2).join(''))
  watcher.poll()
  assert.equal(events.length, 5)
  assert.equal(watcher.getSessions()[0].status, 'completed')
})

test('holds a partially written line until it is finished', () => {
  const [first, second] = RUN
  fs.writeFileSync(file, first + second.slice(0, 20))
  watcher.start()
  assert.equal(events.length, 1)

  fs.appendFileSync(file, second.slice(20))
  watcher.poll()
  assert.equal(events.length, 2)
  assert.equal(events[1].payload.content, 'Write a report\non churn')
})

test('skips malformed and non-event lines', () => {
  fs.writeFileSync(file, 'not json\n{"foo":1}\n' + RUN[0] + '{"type":"message"}\n')
  watcher.start()
  assert.deepEqual(events.map(e => e.type), ['agent_spawn'])
})

test('truncating the file starts a new session', () => {
  fs.writeFileSync(file, RUN.join(''))
  watcher.start()
  const firstId = events[0].sessionId

  fs.writeFileSync(file, '')
  watcher.poll()
  fs.writeFileSync(file, RUN.slice(0, 2).join(''))
  watcher.poll()

  const secondId = events[events.length - 1].sessionId
  assert.notEqual(secondId, firstId)
  assert.match(secondId!, /-2$/)
  assert.equal(lifecycle.filter(l => l.startsWith('started')).length, 2)
  assert.deepEqual(watcher.getSessions().map(s => [s.id, s.status]), [[secondId, 'active']])
})

test('a rewrite that outgrows the old file between polls still starts a new session', () => {
  fs.writeFileSync(file, RUN.slice(0, 2).join(''))
  watcher.start()
  const firstId = events[0].sessionId

  // New run with different timestamps, already longer than the old file.
  const rerun = [
    line(0, 'agent_spawn', { name: 'supervisor', isMain: true, task: 'Write a report' }),
    line(0.002, 'message', { agent: 'supervisor', role: 'user', content: 'Write a report\non churn' }),
    line(0.015, 'tool_call_start', { agent: 'supervisor', tool: 'search', args: 'q' }),
  ]
  fs.writeFileSync(file, rerun.join(''))
  watcher.poll()

  const newEvents = events.filter(e => e.sessionId !== firstId)
  assert.deepEqual(newEvents.map(e => e.type), ['agent_spawn', 'message', 'tool_call_start'])
  assert.deepEqual(lifecycle.map(l => l.split(':')[0]), ['started', 'updated', 'ended', 'started', 'updated'])
})

test('a repeat invocation reactivates the completed session', () => {
  fs.writeFileSync(file, RUN.join(''))
  watcher.start()
  fs.appendFileSync(file, line(5, 'agent_spawn', { name: 'supervisor', isMain: true }))
  watcher.poll()

  assert.equal(new Set(events.map(e => e.sessionId)).size, 1)
  assert.deepEqual(lifecycle, [
    'started:1:events.jsonl', 'updated:1:Write a report', 'ended:1:Write a report', 'started:1:Write a report',
  ])
  assert.equal(watcher.getSessions()[0].status, 'active')
})

test('parseEventLogPaths splits on the platform delimiter and resolves paths', () => {
  assert.deepEqual(parseEventLogPaths(undefined), [])
  assert.deepEqual(parseEventLogPaths(''), [])
  const value = ['/tmp/a.jsonl', ' rel/b.jsonl ', ''].join(path.delimiter)
  assert.deepEqual(parseEventLogPaths(value), ['/tmp/a.jsonl', path.resolve('rel/b.jsonl')])
})
