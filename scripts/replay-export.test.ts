import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import * as vm from 'vm'

import { buildReplayHtml, normalizeEvents, replayFileName, startEmbeddedReplay, type ReplayData } from '../web/lib/replay-export'

const events = [
  { time: 105.5, type: 'agent_spawn', payload: { name: 'lead', isMain: true } },
  { time: 100, type: 'message', payload: { agent: 'lead', role: 'user', content: 'Close the tag: </script><script>alert(1)</script> & <b>' } },
  { time: 106, type: 'agent_complete', payload: { name: 'lead' } },
  { time: NaN, type: 'bad', payload: {} },
]

test('events are made relative to the first one, sorted, and cleaned', () => {
  const out = normalizeEvents(events)
  assert.deepEqual(out.map(e => [e.time, e.type]), [[0, 'message'], [5.5, 'agent_spawn'], [6, 'agent_complete']])
  assert.deepEqual(normalizeEvents([]), [])
})

test('the page embeds data and bundle safely, and the data round-trips', () => {
  const js = 'window.__ran = "</script> in a string";'
  const html = buildReplayHtml({ js, css: 'body{color:red}</style>', label: 'Audit <b>billing</b>', events, sessionId: 's1', exportedAt: new Date('2026-09-27T12:00:00Z') })
  // Nothing can close the script or style elements early
  const scripts = html.split('<script>').length - 1
  assert.equal(scripts, 2)
  assert.equal(html.split('</script>').length - 1, 2)
  assert.equal(html.split('</style>').length - 1, 2)
  assert.ok(html.includes('<title>Audit &lt;b&gt;billing&lt;/b&gt; · Agent Flow replay</title>'))
  assert.ok(html.includes('exported 2026-09-27 12:00 UTC'))

  // Run the data script: the events come back intact
  const dataScript = html.slice(html.indexOf('<script>window.__AGENT_FLOW_REPLAY__') + 8, html.indexOf('</script>'))
  const sandbox: { window: { __AGENT_FLOW_REPLAY__?: ReplayData } } = { window: {} }
  vm.runInNewContext(dataScript, sandbox)
  const data = sandbox.window.__AGENT_FLOW_REPLAY__!
  assert.equal(data.session.label, 'Audit <b>billing</b>')
  assert.equal(data.events.length, 3)
  assert.equal(data.events[0].payload.content, 'Close the tag: </script><script>alert(1)</script> & <b>')

  // And the bundle still runs, with its string unchanged
  const bundle = html.slice(html.lastIndexOf('<script>') + 8, html.lastIndexOf('</script>'))
  const ctx: { window: { __ran?: string } } = { window: {} }
  vm.runInNewContext(bundle, ctx)
  assert.equal(ctx.window.__ran, '</script> in a string')
})

test('the embedded replay posts the session, then events at their pace', async () => {
  const data: ReplayData = {
    version: 1, session: { id: 's1', label: 'demo' }, exportedAt: '',
    events: normalizeEvents([
      { time: 0, type: 'agent_spawn', payload: { name: 'a', isMain: true } },
      { time: 0.005, type: 'message', payload: { agent: 'a', role: 'user', content: 'hi' } },
      { time: 0.3, type: 'agent_complete', payload: { name: 'a' } },
    ]),
  }
  const posted: { type: string; at: number; events?: { type: string; sessionId: string }[] }[] = []
  const started = Date.now()
  await new Promise<void>(resolve => {
    startEmbeddedReplay(data, (m: unknown) => {
      const msg = m as { type: string; events?: { type: string; sessionId: string }[] }
      posted.push({ ...msg, at: Date.now() - started })
      if (msg.type === 'session-ended') resolve()
    })
  })
  assert.deepEqual(posted.map(p => p.type), ['session-list', 'agent-event-batch', 'agent-event-batch', 'session-ended'])
  assert.deepEqual(posted[1].events!.map(e => [e.type, e.sessionId]), [['agent_spawn', 's1'], ['message', 's1']])
  assert.ok(posted[2].at - posted[1].at >= 250, 'the pause between events is kept')
})

test('file names', () => {
  assert.equal(replayFileName('Triage this alert: SEV1 checkout API!', 'html'), 'agent-flow-triage-this-alert-sev1-checkout-api.html')
  assert.equal(replayFileName('***', 'jsonl'), 'agent-flow-session.jsonl')
})
