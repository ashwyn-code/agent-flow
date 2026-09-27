#!/usr/bin/env node
/**
 * Convert OpenTelemetry traces (an OTLP JSON file, or JSON Lines of them as
 * the Collector's file exporter writes) into Agent Flow JSONL event logs.
 *
 *   pnpm otel:import traces.json                 # list the traces in the file
 *   pnpm otel:import traces.json --out traces/   # one <trace id>.jsonl per trace
 *   pnpm otel:import traces.json --trace 4bf9 > run.jsonl
 *
 * View a result with `agent-flow-app --event-log run.jsonl`, or skip this
 * step and pass the OTLP file itself with `--otel-file traces.json`.
 */
import * as fs from 'fs'
import * as path from 'path'

import { convertSpans } from './otel/convert'
import { readOtlpFile } from './otel/receiver'

function usage(code: number): never {
  console.error('Usage: otel-import <otlp.json|otlp.jsonl> [--out <dir>] [--trace <id prefix>]')
  process.exit(code)
}

function main(argv: string[]) {
  let input: string | undefined
  let outDir: string | undefined
  let trace: string | undefined
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--out' && argv[i + 1]) outDir = argv[++i]
    else if (arg === '--trace' && argv[i + 1]) trace = argv[++i].toLowerCase()
    else if (arg === '-h' || arg === '--help') usage(0)
    else if (!arg.startsWith('-') && !input) input = arg
    else usage(1)
  }
  if (!input) usage(1)

  const traces = convertSpans(readOtlpFile(input))
  if (!traces.length) {
    console.error(`No agent traces found in ${input}.`)
    process.exit(1)
  }
  const toJsonl = (events: { time: number; type: string; payload: unknown }[]) =>
    events.map(e => JSON.stringify({ time: e.time, type: e.type, payload: e.payload })).join('\n') + '\n'

  if (trace) {
    const matches = traces.filter(t => t.traceId.startsWith(trace!))
    if (matches.length !== 1) {
      console.error(matches.length ? `"${trace}" matches ${matches.length} traces; give more of the id.` : `No trace id starts with "${trace}".`)
      process.exit(1)
    }
    process.stdout.write(toJsonl(matches[0].events))
    return
  }

  if (outDir) {
    fs.mkdirSync(outDir, { recursive: true })
    for (const t of traces) {
      const file = path.join(outDir, `${t.traceId}.jsonl`)
      fs.writeFileSync(file, toJsonl(t.events))
      console.log(`${file}  ${t.name}  ${t.events.length} events, ${t.duration.toFixed(1)}s`)
    }
    return
  }

  for (const t of traces) {
    const agents = t.events.filter(e => e.type === 'agent_spawn').length
    console.log(`${t.traceId}  ${t.name}  ${agents} agents, ${t.events.length} events, ${t.duration.toFixed(1)}s`)
  }
  console.log(`\nWrite them with --out <dir>, or one with --trace <id prefix>.`)
}

main(process.argv.slice(2))
