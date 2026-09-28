#!/usr/bin/env node
/**
 * Build a self-contained HTML replay from an Agent Flow JSONL event log (or
 * an OTLP JSON trace file), for CI artifacts, pull requests and incident
 * write-ups. Open the result in any browser; nothing else is needed.
 *
 *   pnpm build:app                                  # once: builds the UI bundle
 *   pnpm replay:export run.jsonl -o run.html
 *   pnpm replay:export traces.otlp.json --trace 4bf9 -o incident.html
 */
import * as fs from 'fs'
import * as path from 'path'

import { buildReplayHtml } from '../web/lib/replay-export'
import { convertSpans } from './otel/convert'
import { spansFromText } from './otel/otlp'

const WEBVIEW = path.join(__dirname, '..', 'app', 'dist', 'webview')

function usage(code: number): never {
  console.error('Usage: replay-export <events.jsonl | otlp.json> [-o out.html] [--label <text>] [--trace <id prefix>]')
  process.exit(code)
}

function readEvents(text: string, trace?: string): { events: { time: number; type: string; payload: unknown }[]; label?: string } {
  // An OpenTelemetry export: convert one trace
  if (/"resourceSpans"|"resource_spans"/.test(text.slice(0, 4096))) {
    const traces = convertSpans(spansFromText(text))
    const matches = trace ? traces.filter(t => t.traceId.startsWith(trace)) : traces
    if (matches.length !== 1) {
      console.error(matches.length ? `${matches.length} traces in the file; pick one with --trace:` : 'No matching agent trace in the file.')
      for (const t of traces) console.error(`  ${t.traceId}  ${t.name}`)
      process.exit(1)
    }
    return { events: matches[0].events, label: matches[0].name }
  }
  const events = []
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue
    try {
      const e = JSON.parse(line)
      if (typeof e.time === 'number' && typeof e.type === 'string') events.push({ time: e.time, type: e.type, payload: e.payload ?? {} })
    } catch { /* skip */ }
  }
  const prompt = events.find(e => e.type === 'message' && (e.payload as { role?: string }).role === 'user')
  const label = typeof (prompt?.payload as { content?: unknown })?.content === 'string'
    ? String((prompt!.payload as { content: string }).content).split('\n')[0].slice(0, 60)
    : undefined
  return { events, label }
}

function main(argv: string[]) {
  let input: string | undefined
  let out: string | undefined
  let label: string | undefined
  let trace: string | undefined
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if ((arg === '-o' || arg === '--out') && argv[i + 1]) out = argv[++i]
    else if (arg === '--label' && argv[i + 1]) label = argv[++i]
    else if (arg === '--trace' && argv[i + 1]) trace = argv[++i].toLowerCase()
    else if (arg === '-h' || arg === '--help') usage(0)
    else if (!arg.startsWith('-') && !input) input = arg
    else usage(1)
  }
  if (!input) usage(1)

  const jsPath = path.join(WEBVIEW, 'index.js')
  if (!fs.existsSync(jsPath)) {
    console.error(`The UI bundle isn't built yet (${path.relative(process.cwd(), jsPath)}). Run \`pnpm build:app\` first.`)
    process.exit(1)
  }
  const { events, label: found } = readEvents(fs.readFileSync(input, 'utf8'), trace)
  if (!events.length) {
    console.error(`No events in ${input}.`)
    process.exit(1)
  }
  const html = buildReplayHtml({
    js: fs.readFileSync(jsPath, 'utf8'),
    css: fs.existsSync(path.join(WEBVIEW, 'index.css')) ? fs.readFileSync(path.join(WEBVIEW, 'index.css'), 'utf8') : '',
    label: label ?? found ?? path.basename(input),
    events,
  })
  const target = out ?? input.replace(/\.(jsonl|json|ndjson)(\.gz)?$/i, '') + '.replay.html'
  fs.writeFileSync(target, html)
  console.log(`${target}  ${events.length} events, ${(html.length / 1024).toFixed(0)} KB`)
}

main(process.argv.slice(2))
