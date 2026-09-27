import { DEFAULT_RELAY_PORT } from '../../extension/src/constants'

/** Parse CLI arguments. Keeps it simple — no dependencies. */
export function parseArgs(argv: string[]) {
  let port = DEFAULT_RELAY_PORT
  let open = true
  let verbose = false
  const eventLogs: string[] = []
  let ingestHost: string | undefined
  let ingestPort: number | undefined
  let ingestToken: string | undefined = process.env.AGENT_FLOW_INGEST_TOKEN || undefined

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if ((arg === '--port' || arg === '-p') && argv[i + 1]) {
      const n = parseInt(argv[i + 1], 10)
      if (!isNaN(n) && n > 0 && n < 65536) port = n
      i++
    } else if ((arg === '--event-log' || arg === '-e') && argv[i + 1]) {
      eventLogs.push(argv[i + 1])
      i++
    } else if (arg === '--ingest-host' && argv[i + 1]) {
      ingestHost = argv[++i]
    } else if (arg === '--ingest-port' && argv[i + 1]) {
      const n = parseInt(argv[++i], 10)
      if (!isNaN(n) && n > 0 && n < 65536) ingestPort = n
    } else if (arg === '--ingest-token' && argv[i + 1]) {
      ingestToken = argv[++i]
    } else if (arg === '--no-open') {
      open = false
    } else if (arg === '--verbose' || arg === '-v') {
      verbose = true
    } else if (arg === '--help' || arg === '-h') {
      console.log(`
Usage: agent-flow [options]

Options:
  -p, --port <number>  Port for the server (default: ${DEFAULT_RELAY_PORT})
  -e, --event-log <path>
                       Also show an Agent Flow JSONL event log, e.g. from the
                       LangGraph adapter (repeatable; AGENT_FLOW_EVENT_LOG also works)
  --ingest-token <token>
                       Require this bearer token on POST /ingest (AGENT_FLOW_INGEST_TOKEN)
  --ingest-host <host>, --ingest-port <port>
                       Also accept POST /ingest on a separate listener, e.g. 0.0.0.0:3101,
                       so agents on other hosts can send events. It serves /ingest only
                       (the UI stays on 127.0.0.1) and requires --ingest-token.
  --no-open            Don't open the browser automatically
  -v, --verbose        Show detailed event logs
  -h, --help           Show this help message
`)
      process.exit(0)
    }
  }

  return { port, open, verbose, eventLogs, ingestHost, ingestPort, ingestToken }
}
