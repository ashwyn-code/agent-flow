/**
 * Combined HTTP server: serves the visualizer UI and streams events via SSE.
 * Reuses the extension's hook server, transcript parser, and session watcher.
 */
import * as http from 'http'
import * as os from 'os'
import * as path from 'path'
import { exec, execFile } from 'child_process'

import { createRelay } from '../../scripts/relay'
import { createTelemetryClient } from '../../scripts/telemetry'
import { serveStatic } from './static'

interface ServerOptions {
  port: number
  openBrowser: boolean
  workspace: string
  verbose?: boolean
  /** Event logs from --event-log; falls back to AGENT_FLOW_EVENT_LOG when empty. */
  eventLogs?: string[]
  /** Separate listener that accepts only POST /ingest (for agents on other hosts) */
  ingestHost?: string
  ingestPort?: number
  ingestToken?: string
}

function isLoopbackHost(host: string): boolean {
  return host === '127.0.0.1' || host === 'localhost' || host === '::1' || host.startsWith('127.')
}

export async function startServer(options: ServerOptions) {
  const { port, openBrowser, workspace } = options

  const configDir = path.join(os.homedir(), '.agent-flow')
  const telemetry = createTelemetryClient({
    logDir: path.join(configDir, 'telemetry'),
    installIdPath: path.join(configDir, 'installation-id'),
  })
  await telemetry.init()

  const eventLogs = options.eventLogs?.length ? options.eventLogs.map(p => path.resolve(p)) : undefined
  if (options.ingestHost && !isLoopbackHost(options.ingestHost) && !options.ingestToken) {
    console.error('Refusing to accept events from other hosts without a token: pass --ingest-token or set AGENT_FLOW_INGEST_TOKEN.')
    process.exit(1)
  }
  const relay = await createRelay({ workspace, verbose: options.verbose, telemetry, eventLogs, ingestToken: options.ingestToken })

  const server = http.createServer((req, res) => {
    // SSE endpoint
    if (req.url === '/events') {
      return relay.handleSSE(req, res)
    }

    // Events from adapters' HTTP transport (loopback, or with a token)
    if (req.url === '/ingest') {
      return relay.handleIngest(req, res)
    }

    // Static files (UI)
    if (req.method === 'GET') {
      return serveStatic(req, res)
    }

    res.writeHead(404)
    res.end('Not found')
  })

  // Optional network listener that serves nothing but /ingest
  let ingestServer: http.Server | null = null
  if (options.ingestHost || options.ingestPort) {
    const ingestHost = options.ingestHost ?? '127.0.0.1'
    const ingestPort = options.ingestPort ?? port + 100
    ingestServer = http.createServer((req, res) => {
      if (req.url === '/ingest') return relay.handleIngest(req, res)
      res.writeHead(404)
      res.end('Not found')
    })
    ingestServer.listen(ingestPort, ingestHost, () => {
      console.log(`Accepting events at http://${ingestHost}:${ingestPort}/ingest${options.ingestToken ? ' (token required)' : ''}`)
    })
  }

  server.listen(port, '127.0.0.1', () => {
    const url = `http://127.0.0.1:${port}`
    console.log(`Server running at ${url}`)
    console.log('Waiting for agent events...\n')

    if (openBrowser) {
      openURL(url)
    }
  })

  // Cleanup on exit. Idempotent — repeat signals (Ctrl+C spam, SIGTERM+SIGHUP,
  // etc.) would otherwise emit duplicate session_end events and race the
  // telemetry sync loop against itself.
  let shuttingDown = false
  function cleanup() {
    if (shuttingDown) return
    shuttingDown = true
    server.close()
    ingestServer?.close()
    relay.dispose()
    void telemetry.dispose().finally(() => process.exit(0))
  }
  process.on('SIGINT', cleanup)
  process.on('SIGTERM', cleanup)
  // SIGHUP fires when the controlling terminal closes (SSH session drops, tmux
  // pane killed). Without a handler, Node's default behavior is to terminate
  // without running cleanup — so session_end never flushes.
  process.on('SIGHUP', cleanup)
}

function openURL(url: string) {
  if (process.platform === 'win32') {
    // 'start' is a shell builtin on Windows — must use exec, not execFile
    exec(`start "" "${url}"`)
  } else {
    const cmd = process.platform === 'darwin' ? 'open' : 'xdg-open'
    execFile(cmd, [url], () => {})
  }
}
