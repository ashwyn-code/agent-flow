/**
 * One trace's spans → Agent Flow events.
 *
 * Understands the two span vocabularies agent frameworks emit:
 *   - OpenTelemetry GenAI semantic conventions (`gen_ai.operation.name`:
 *     invoke_agent, chat, execute_tool, …) — Strands, Microsoft Agent
 *     Framework, Google ADK and others;
 *   - OpenInference (`openinference.span.kind`: AGENT, LLM, TOOL, CHAIN, …) —
 *     Arize Phoenix's instrumentations for LangChain/LangGraph, OpenAI Agents,
 *     CrewAI and others.
 *
 * Agents become agents, sub-agents (nested, or called as tools) become
 * sub-agents, LLM calls become model/thinking/message/context events, and
 * tool calls become tool calls. A span that orchestrates other agents without
 * calling a model itself (a graph, swarm, workflow, or a sequential / parallel
 * / loop agent) is drawn as a graph: its children are the nodes, and the
 * edges are the order they actually ran in, taken from span timing (a node
 * follows the nodes that finished just before it started). When a framework
 * records its declared graph (Agent Framework's `workflow.build` span), the
 * declared edges, including conditional ones, are drawn too.
 */
import type { AgentEvent } from '../../extension/src/protocol'
import type { Attributes, Span } from './otlp'

const MAX_TEXT = 2000
const MAX_ARGS = 1000
const MAX_TASK = 300

const LLM_OPS = new Set(['chat', 'text_completion', 'generate_content', 'completion', 'embeddings'])
const WORKFLOW_OPS = new Set(['invoke_graph', 'invoke_swarm', 'invoke_workflow', 'invoke_multiagent'])
const START = '__start__'
const END = '__end__'

// ─── Declared graphs ────────────────────────────────────────────────────────

export interface DeclaredGraph {
  start?: string
  nodes: string[]
  edges: { source: string; target: string; conditional: boolean; label?: string }[]
}

/** Declared workflow graphs, keyed by workflow id. They can arrive in another
 *  trace than the run (Agent Framework builds a workflow in its own trace), so
 *  callers keep one of these across traces. */
export type GraphRegistry = Map<string, DeclaredGraph>

/** Record any declared graphs found in these spans. */
export function collectDeclaredGraphs(spans: Span[], registry: GraphRegistry): void {
  for (const span of spans) {
    const definition = str(span.attributes['workflow.definition'])
    if (!definition) continue
    const parsed = parseJson(definition)
    const id = str(span.attributes['workflow.id']) || str(parsed?.id)
    if (!id || !parsed || !Array.isArray(parsed.edge_groups)) continue
    const graph: DeclaredGraph = { start: str(parsed.start_executor_id) || undefined, nodes: [], edges: [] }
    const nodes = new Set<string>()
    if (parsed.executors && typeof parsed.executors === 'object') {
      for (const id of Object.keys(parsed.executors)) nodes.add(id)
    }
    for (const group of parsed.edge_groups) {
      const type = str(group?.type)
      if (type === 'InternalEdgeGroup') continue
      const labels = new Map<string, string>()
      for (const c of Array.isArray(group?.cases) ? group.cases : []) {
        const target = str(c?.target_id)
        if (target) labels.set(target, str(c?.condition_name) || (str(c?.type) === 'Default' ? 'default' : ''))
      }
      const conditional = type === 'SwitchCaseEdgeGroup' || !!str(group?.selection_func_name) || labels.size > 0
      for (const e of Array.isArray(group?.edges) ? group.edges : []) {
        const source = str(e?.source_id)
        const target = str(e?.target_id)
        if (!source || !target || source.startsWith('internal:')) continue
        nodes.add(source)
        nodes.add(target)
        const raw = labels.get(target) || str(e?.condition_name)
        const label = raw && !raw.startsWith('<') ? raw : undefined  // not '<lambda>'
        graph.edges.push({ source, target, conditional: conditional || !!raw, ...(label ? { label } : {}) })
      }
    }
    graph.nodes = [...nodes]
    registry.set(id, graph)
  }
}

// ─── Attribute helpers ──────────────────────────────────────────────────────

function str(v: unknown): string {
  return typeof v === 'string' ? v : typeof v === 'number' || typeof v === 'boolean' ? String(v) : ''
}

function num(v: unknown): number | undefined {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN
  return Number.isFinite(n) ? n : undefined
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseJson(v: unknown): any {
  if (typeof v !== 'string') return v && typeof v === 'object' ? v : undefined
  const t = v.trim()
  if (!t || (t[0] !== '{' && t[0] !== '[')) return undefined
  try { return JSON.parse(t) } catch { return undefined }
}

function clip(text: string, max = MAX_TEXT): string {
  const t = text.trim()
  return t.length > max ? t.slice(0, max - 1) + '…' : t
}

/** OpenInference flattens lists into `prefix.0.suffix` keys; collect them. */
function indexed(attrs: Attributes, prefix: string): Map<number, Attributes> {
  const out = new Map<number, Attributes>()
  const p = prefix + '.'
  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith(p)) continue
    const rest = key.slice(p.length)
    const dot = rest.indexOf('.')
    const i = Number(dot < 0 ? rest : rest.slice(0, dot))
    if (!Number.isInteger(i)) continue
    const entry = out.get(i) ?? {}
    entry[dot < 0 ? '' : rest.slice(dot + 1)] = value
    out.set(i, entry)
  }
  return new Map([...out.entries()].sort((a, b) => a[0] - b[0]))
}

interface Parts { text: string[]; reasoning: string[] }

/** Pull text and reasoning out of the many message shapes in the wild:
 *  GenAI `parts`, Bedrock/Strands content blocks, OpenAI Responses items,
 *  LangChain messages and generations, Gemini content parts, plain strings. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function partsOf(value: any, into: Parts = { text: [], reasoning: [] }, depth = 0): Parts {
  if (value == null || depth > 8) return into
  if (typeof value === 'string') {
    const parsed = parseJson(value)
    if (parsed !== undefined) return partsOf(parsed, into, depth + 1)
    if (value.trim()) into.text.push(value.trim())
    return into
  }
  if (Array.isArray(value)) {
    for (const v of value) partsOf(v, into, depth + 1)
    return into
  }
  if (typeof value !== 'object') return into
  const type = str(value.type)
  if (type === 'tool_call' || type === 'function_call' || type === 'tool_use' || type === 'tool_call_response'
    || value.function_call || value.toolUse || value.functionCall || value.function_response || value.toolResult) {
    return into
  }
  if (type === 'reasoning' || type === 'thinking') {
    const r = partsOf(value.content ?? value.text ?? value.thinking ?? value.summary, undefined, depth + 1)
    into.reasoning.push(...r.text, ...r.reasoning)
    return into
  }
  if (value.reasoningContent) {
    const text = str(value.reasoningContent?.reasoningText?.text)
    if (text) into.reasoning.push(text)
    return into
  }
  if (value.thought === true && typeof value.text === 'string') {
    into.reasoning.push(value.text)
    return into
  }
  if (typeof value.text === 'string' && (type === '' || type === 'text' || type === 'output_text' || type === 'input_text' || type === 'summary_text' || type === 'ChatGeneration')) {
    if (type === 'ChatGeneration' && value.message) return partsOf(value.message, into, depth + 1)
    if (value.text.trim()) into.text.push(value.text.trim())
    return into
  }
  if (value.generations) return partsOf(value.generations, into, depth + 1)
  if (value.kwargs) return partsOf(value.kwargs.content, into, depth + 1)
  if ('content' in value) return partsOf(value.content, into, depth + 1)
  if ('parts' in value) return partsOf(value.parts, into, depth + 1)
  if ('message' in value) return partsOf(value.message, into, depth + 1)
  if ('messages' in value) return partsOf(value.messages, into, depth + 1)
  return into
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function messageRole(m: any): string {
  if (Array.isArray(m) && typeof m[0] === 'string') return m[0] === 'human' ? 'user' : m[0]
  const role = str(m?.role) || str(m?.type) || str(m?.kwargs?.type)
  if (role === 'human') return 'user'
  if (!role && Array.isArray(m?.id) && str(m.id[m.id.length - 1]) === 'HumanMessage') return 'user'
  return role
}

/** User messages sent in this span (in order). */
function userMessages(span: Span): string[] {
  const a = span.attributes
  const out: string[] = []
  const push = (m: unknown) => {
    const text = partsOf(m).text.join('\n').trim()
    if (text) out.push(text)
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fromList = (list: any) => {
    if (!Array.isArray(list)) return
    for (const m of list.flat()) {
      if (messageRole(m) === 'user') push(Array.isArray(m) ? m[1] : m.parts ?? m.content ?? m.kwargs?.content ?? m.text)
    }
  }
  fromList(parseJson(a['gen_ai.input.messages']))
  if (!out.length) {
    for (const m of indexed(a, 'llm.input_messages').values()) {
      if (str(m['message.role']) === 'user') push(m['message.content'])
    }
  }
  if (!out.length) {
    for (const e of span.events) {
      if (e.name === 'gen_ai.user.message') push(e.attributes.content)
    }
  }
  if (!out.length) {
    const llmRequest = parseJson(a['gcp.vertex.agent.llm_request'])
    fromList(llmRequest?.contents)
  }
  if (!out.length) {
    const input = parseJson(a['input.value'])
    if (Array.isArray(input)) fromList(input)
    else if (input && Array.isArray(input.messages)) fromList(input.messages)
    else if (input === undefined && str(a['input.value']) && isAgentLike(span)) push(str(a['input.value']))
  }
  if (!out.length) {
    const prompt = parseJson(a['gen_ai.prompt'])
    if (Array.isArray(prompt)) fromList(prompt)
    else if (str(a['gen_ai.prompt'])) push(str(a['gen_ai.prompt']))
  }
  return out
}

/** What this span said: its reply text and any reasoning. */
function outputOf(span: Span): Parts {
  const a = span.attributes
  const genai = parseJson(a['gen_ai.output.messages'])
  if (genai) return partsOf(genai)
  const choices = span.events.filter(e => e.name === 'gen_ai.choice')
  if (choices.length) return partsOf(choices.map(e => e.attributes.message ?? e.attributes.content))
  const oi = indexed(a, 'llm.output_messages')
  if (oi.size) {
    const parts: Parts = { text: [], reasoning: [] }
    for (const m of oi.values()) {
      if (str(m['message.content'])) parts.text.push(str(m['message.content']))
      for (const c of indexed(m as Attributes, 'message.contents').values()) {
        const type = str(c['message_content.type'])
        const text = str(c['message_content.text'])
        if (text) (type === 'reasoning' || type === 'thinking' ? parts.reasoning : parts.text).push(text)
      }
    }
    // Reasoning is only in the raw output for some integrations
    if (!parts.reasoning.length) parts.reasoning = partsOf(parseJson(a['output.value'])).reasoning
    return parts
  }
  const vertex = parseJson(a['gcp.vertex.agent.llm_response'])
  if (vertex) return partsOf(vertex.content ?? vertex)
  if (a['output.value'] !== undefined) return partsOf(a['output.value'])
  if (a['gen_ai.completion'] !== undefined) return partsOf(a['gen_ai.completion'])
  return { text: [], reasoning: [] }
}

function toolArgs(span: Span): string {
  const a = span.attributes
  const direct = a['gen_ai.tool.call.arguments'] ?? a['gcp.vertex.agent.tool_call_args'] ?? a['input.value']
  if (direct !== undefined) return typeof direct === 'string' ? direct : JSON.stringify(direct)
  const event = span.events.find(e => e.name === 'gen_ai.tool.message')
  return event ? str(event.attributes.content) : ''
}

function toolResult(span: Span): string {
  const a = span.attributes
  const direct = a['gen_ai.tool.call.result'] ?? a['gcp.vertex.agent.tool_response'] ?? a['output.value']
  if (direct !== undefined) {
    const parsed = parseJson(direct)
    // LangChain ToolMessage: {"content": "...", "type": "tool", ...}
    if (parsed && !Array.isArray(parsed) && parsed.type === 'tool' && 'content' in parsed) return str(parsed.content) || JSON.stringify(parsed.content)
    return typeof direct === 'string' ? direct : JSON.stringify(direct)
  }
  const choice = span.events.find(e => e.name === 'gen_ai.choice')
  if (choice) return partsOf(choice.attributes.message).text.join('\n')
  return ''
}

function errorOf(span: Span): string | undefined {
  if (span.statusCode !== 2) return undefined
  const exception = span.events.find(e => e.name === 'exception')
  const message = span.statusMessage || str(exception?.attributes['exception.message'])
    || str(exception?.attributes['exception.type']) || str(span.attributes['error.type'])
  // Some instrumentations append the whole traceback to the status message
  const first = message.split(/Traceback \(most recent call last\)/)[0].trim().split(/\r?\n/)[0]
  return first || 'error'
}

/** Tools that report failure in their result instead of raising, e.g.
 *  Google ADK's {"status": "error", "error_message": "..."} convention. */
function resultError(result: string): string | undefined {
  const parsed = parseJson(result)
  if (!parsed || Array.isArray(parsed) || str(parsed.status).toLowerCase() !== 'error') return undefined
  return str(parsed.error_message) || str(parsed.error) || str(parsed.message) || 'error'
}

function modelOf(span: Span): string {
  const a = span.attributes
  return str(a['gen_ai.response.model']) || str(a['gen_ai.request.model']) || str(a['llm.model_name'])
}

function tokensOf(span: Span): number | undefined {
  const a = span.attributes
  const input = num(a['gen_ai.usage.input_tokens']) ?? num(a['gen_ai.usage.prompt_tokens']) ?? num(a['llm.token_count.prompt'])
  const output = num(a['gen_ai.usage.output_tokens']) ?? num(a['gen_ai.usage.completion_tokens']) ?? num(a['llm.token_count.completion'])
  const total = num(a['gen_ai.usage.total_tokens']) ?? num(a['llm.token_count.total'])
  if (input === undefined && output === undefined) return total
  return (input ?? 0) + (output ?? 0)
}

function langgraphMeta(span: Span): { node: string; step?: number } | null {
  const meta = parseJson(span.attributes.metadata)
  const node = str(meta?.langgraph_node)
  return node ? { node, step: num(meta?.langgraph_step) } : null
}

// ─── Span roles ─────────────────────────────────────────────────────────────

type Role = 'llm' | 'tool' | 'guardrail' | 'agent' | 'workflow' | 'chain' | 'other' | 'skip'

function spanKind(span: Span): string {
  return str(span.attributes['openinference.span.kind']).toUpperCase()
}

function isAgentLike(span: Span): boolean {
  return spanKind(span) === 'AGENT' || str(span.attributes['gen_ai.operation.name']) === 'invoke_agent'
}

function spanRole(span: Span): Role {
  const a = span.attributes
  const op = str(a['gen_ai.operation.name'])
  const kind = spanKind(span)
  if (str(a['gen_ai.tool.name']) === '(merged tools)') return 'skip'
  if (kind === 'LLM' || LLM_OPS.has(op) || span.name === 'call_llm') return 'llm'
  if (kind === 'TOOL' || op === 'execute_tool') return 'tool'
  if (kind === 'GUARDRAIL') return 'guardrail'
  if (kind === 'AGENT' || op === 'invoke_agent') return 'agent'
  if (WORKFLOW_OPS.has(op) || span.name === 'workflow.run' || span.name.startsWith('workflow.run ')) return 'workflow'
  if (kind === 'CHAIN') return 'chain'
  if (kind === 'EMBEDDING' || kind === 'RETRIEVER' || kind === 'RERANKER') return 'tool'
  return 'other'
}

/** Graph node id when this span is a node of a graph, else null. */
function graphNodeId(span: Span, parent: Span | undefined): string | null {
  const a = span.attributes
  const executor = str(a['executor.id'])
  if (executor || span.name.startsWith('executor.process')) return executor || span.name.replace(/^executor\.process\s*/, '') || null
  const lg = langgraphMeta(span)
  if (lg && lg.node === span.name) {
    const parentLg = parent ? langgraphMeta(parent) : null
    if (!parentLg || parentLg.node !== span.name) return lg.node
  }
  return null
}

function agentName(span: Span): string {
  const a = span.attributes
  const op = str(a['gen_ai.operation.name'])
  const named = str(a['gen_ai.agent.name']) || str(a['agent.name']) || str(a['workflow.name'])
  if (named) return named
  if (op === 'invoke_graph') return 'graph'
  if (op === 'invoke_swarm') return 'swarm'
  return span.name.replace(/^(invoke_agent|invoke_workflow|workflow\.run)\s+/, '') || 'agent'
}

function toolName(span: Span): string {
  const a = span.attributes
  return str(a['gen_ai.tool.name']) || str(a['tool.name']) || span.name.replace(/^execute_tool\s+/, '') || 'tool'
}

// ─── Logical tree ───────────────────────────────────────────────────────────

interface Timed { span: Span; start: number; end: number }

interface RawNode extends Timed {
  children: RawNode[]
  parent?: RawNode
}

type Item =
  | { t: 'llm'; n: RawNode; merged: Span[] }
  | { t: 'tool'; n: RawNode; inner: Item[] }
  | { t: 'guardrail'; n: RawNode }
  | { t: 'actor'; a: Actor }
  | { t: 'gnode'; n: RawNode; id: string; inner: Item[] }

interface Actor extends Timed {
  base: string
  kind: 'agent' | 'workflow'
  items: Item[]
  /** Node id this actor has in its parent's graph (if the parent is a graph) */
  nodeId?: string
}

function build(node: RawNode): Item[] {
  const role = spanRole(node.span)
  const nodeId = graphNodeId(node.span, node.parent?.span)
  if (nodeId !== null && role !== 'llm' && role !== 'tool') {
    if (role === 'agent' || role === 'workflow') {
      return [{ t: 'actor', a: { ...timed(node), base: nodeId, kind: role, items: collect(node), nodeId } }]
    }
    return [{ t: 'gnode', n: node, id: nodeId, inner: collect(node) }]
  }
  switch (role) {
    case 'skip': return []
    case 'llm': return [{ t: 'llm', n: node, merged: nestedLlms(node) }, ...collect(node).filter(i => i.t !== 'llm')]
    case 'tool': return [{ t: 'tool', n: node, inner: collect(node) }]
    case 'guardrail': return [{ t: 'guardrail', n: node }]
    case 'agent':
    case 'workflow':
      return [{ t: 'actor', a: { ...timed(node), base: agentName(node.span), kind: role, items: collect(node) } }]
    case 'chain':
    case 'other': {
      const inner = collect(node)
      // A chain that runs several agents and no model of its own orchestrates
      // them (e.g. an OpenAI Agents handoff chain); otherwise it's plumbing.
      if (role === 'chain' && countActors(inner) >= 2 && !inner.some(i => i.t === 'llm')) {
        return [{ t: 'actor', a: { ...timed(node), base: agentName(node.span), kind: 'workflow', items: inner } }]
      }
      return inner
    }
  }
}

function timed(n: RawNode): Timed {
  return { span: n.span, start: n.start, end: n.end }
}

function collect(node: RawNode): Item[] {
  return node.children.flatMap(build)
}

function nestedLlms(node: RawNode): Span[] {
  const out: Span[] = []
  const walk = (n: RawNode) => {
    for (const c of n.children) {
      if (spanRole(c.span) === 'llm') out.push(c.span)
      walk(c)
    }
  }
  walk(node)
  return out
}

function countActors(items: Item[]): number {
  return items.filter(i => i.t === 'actor' || i.t === 'gnode').length
}

/** The node's own actor: the one agent it wraps, a new agent named after the
 *  node when it does work itself, or none for a plain function node. */
function actorOfNode(item: Extract<Item, { t: 'gnode' }>): Actor | null {
  const actors = item.inner.filter((i): i is Extract<Item, { t: 'actor' }> => i.t === 'actor')
  const other = item.inner.filter(i => i.t !== 'actor')
  if (actors.length === 1 && other.length === 0) return { ...actors[0].a, nodeId: item.id }
  if (item.inner.length === 0) return null
  return { ...timed(item.n), base: item.id, kind: 'agent', items: item.inner, nodeId: item.id }
}

function isGraph(actor: Actor): boolean {
  if (actor.kind === 'workflow') return true
  if (actor.items.some(i => i.t === 'gnode')) return true
  return !actor.items.some(i => i.t === 'llm') && countActors(actor.items) >= 2
}

// ─── Events ─────────────────────────────────────────────────────────────────

interface Pending { t: number; seq: number; type: AgentEvent['type']; payload: Record<string, unknown> }

export interface ConvertOptions {
  registry?: GraphRegistry
  /** Name for the main agent when the trace has no single top-level agent */
  fallbackName?: string
}

export interface ConvertedTrace {
  traceId: string
  /** Main agent name */
  name: string
  events: AgentEvent[]
  /** Duration in seconds */
  duration: number
}

class Emitter {
  private seq = 0
  readonly out: Pending[] = []
  private readonly names: { name: string; start: number; end: number }[] = []
  private readonly models = new Map<string, string>()

  constructor(private readonly registry: GraphRegistry | undefined) {}

  push(t: number, type: AgentEvent['type'], payload: Record<string, unknown>): void {
    this.out.push({ t, seq: this.seq++, type, payload })
  }

  /** A name unique among agents running at the same time. */
  claim(base: string, start: number, end: number): string {
    let name = base
    for (let n = 2; this.names.some(e => e.name === name && e.start < end && start < e.end); n++) name = `${base} #${n}`
    this.names.push({ name, start, end })
    return name
  }

  actor(actor: Actor, parent: string | null): string {
    const name = this.claim(actor.base, actor.start, actor.end)
    const task = clip(taskOf(actor), MAX_TASK)
    if (parent === null) {
      this.push(actor.start, 'agent_spawn', { name, isMain: true, ...(task ? { task } : {}) })
      const prompt = firstUserMessage(actor)
      if (prompt) this.push(actor.start, 'message', { agent: name, role: 'user', content: clip(prompt) })
    } else {
      this.push(actor.start, 'subagent_dispatch', { parent, child: name, task })
      this.push(actor.start, 'agent_spawn', { name, parent, task })
    }

    if (isGraph(actor)) this.graph(actor, name)
    else this.items(actor.items, name)

    if (parent !== null) {
      this.push(actor.end, 'subagent_return', { parent, child: name, summary: clip(summaryOf(actor), MAX_TASK) })
    }
    this.push(actor.end, 'agent_complete', { name })
    return name
  }

  items(items: Item[], agent: string): void {
    for (const item of items) {
      if (item.t === 'llm') this.llm(item.n, item.merged, agent)
      else if (item.t === 'tool') this.tool(item.n, item.inner, agent)
      else if (item.t === 'guardrail') this.guardrail(item.n, agent)
      else if (item.t === 'actor') this.actor(item.a, agent)
      else {
        const inner = actorOfNode(item)
        if (inner) this.actor(inner, agent)
      }
    }
  }

  llm(n: RawNode, merged: Span[], agent: string): void {
    const spans = [n.span, ...merged]
    const model = spans.map(modelOf).find(Boolean)
    if (model && this.models.get(agent) !== model) {
      this.models.set(agent, model)
      this.push(n.start, 'model_detected', { agent, model })
    }
    const parts = spans.map(outputOf).find(p => p.text.length || p.reasoning.length) ?? { text: [], reasoning: [] }
    if (parts.reasoning.length) this.push(n.end, 'message', { agent, role: 'thinking', content: clip(parts.reasoning.join('\n')) })
    if (parts.text.length) this.push(n.end, 'message', { agent, role: 'assistant', content: clip(parts.text.join('\n')) })
    const tokens = spans.map(tokensOf).find(t => t !== undefined)
    if (tokens !== undefined) this.push(n.end, 'context_update', { agent, tokens })
  }

  tool(n: RawNode, inner: Item[], agent: string): void {
    const tool = toolName(n.span)
    const args = toolArgs(n.span)
    const input = parseJson(args)
    this.push(n.start, 'tool_call_start', {
      agent, tool, args: clip(args, MAX_ARGS),
      ...(input && !Array.isArray(input) ? { inputData: input } : {}),
    })
    this.items(inner, agent)
    const result = toolResult(n.span)
    const error = errorOf(n.span) ?? resultError(result)
    this.push(n.end, 'tool_call_end', {
      agent, tool, result: clip(result || (error ?? '')),
      ...(error ? { isError: true, errorMessage: clip(error, MAX_TASK) } : {}),
    })
  }

  guardrail(n: RawNode, agent: string): void {
    const tool = `guardrail: ${n.span.name}`
    this.push(n.start, 'tool_call_start', { agent, tool, args: '' })
    const error = errorOf(n.span)
    this.push(n.end, 'tool_call_end', {
      agent, tool, result: error ? 'tripped' : 'passed',
      ...(error ? { isError: true, errorMessage: clip(error, MAX_TASK) } : {}),
    })
  }

  graph(actor: Actor, agent: string): void {
    interface Invocation { id: string; start: number; end: number; actor: Actor | null; item: Item; from: string[]; step: number; error?: string }
    const invocations: Invocation[] = []
    const loose: Item[] = []
    for (const item of actor.items) {
      if (item.t === 'actor') {
        invocations.push({ id: item.a.nodeId ?? item.a.base, start: item.a.start, end: item.a.end, actor: item.a, item, from: [], step: 0, error: errorOf(item.a.span) })
      } else if (item.t === 'gnode') {
        invocations.push({ id: item.id, start: item.n.start, end: item.n.end, actor: actorOfNode(item), item, from: [], step: 0, error: errorOf(item.n.span) })
      } else {
        loose.push(item)
      }
    }
    invocations.sort((a, b) => a.start - b.start || a.end - b.end)

    // A node follows the nodes that finished before it started and weren't
    // themselves followed by another node that also finished before it.
    const predecessors = new Set<Invocation>()
    for (const inv of invocations) {
      const before = invocations.filter(p => p !== inv && p.end <= inv.start)
      const preds = before.filter(p => !before.some(q => q !== p && q.start >= p.end && q.end <= inv.start))
      preds.forEach(p => predecessors.add(p))
      inv.from = preds.length ? [...new Set(preds.map(p => p.id))] : [START]
      inv.step = preds.length ? Math.max(...preds.map(p => p.step)) + 1 : 1
    }
    const sinks = invocations.filter(inv => !predecessors.has(inv))
    const endFrom = sinks.length ? [...new Set(sinks.map(s => s.id))] : [START]

    // Structure: what ran, plus the declared graph when the framework recorded one
    const declared = this.registry?.get(str(actor.span.attributes['workflow.id']))
    const nodes = new Map<string, Record<string, unknown>>()
    nodes.set(START, { id: START, label: START, kind: 'start' })
    for (const id of declared?.nodes ?? []) nodes.set(id, { id, label: id, kind: 'node' })
    const edges = new Map<string, Record<string, unknown>>()
    for (const e of declared?.edges ?? []) edges.set(`${e.source}\u0000${e.target}`, { ...e })
    if (declared?.start) edges.set(`${START}\u0000${declared.start}`, { source: START, target: declared.start, conditional: false })
    const childNames = new Map<Invocation, string>()
    for (const inv of invocations) {
      if (!nodes.has(inv.id) || nodes.get(inv.id)!.kind === 'node') {
        nodes.set(inv.id, { id: inv.id, label: inv.id, kind: inv.actor && isGraph(inv.actor) ? 'subgraph' : 'node' })
      }
      for (const source of inv.from) {
        const key = `${source}\u0000${inv.id}`
        if (!edges.has(key)) edges.set(key, { source, target: inv.id, conditional: false })
      }
    }
    for (const source of endFrom) {
      const key = `${source}\u0000${END}`
      if (!edges.has(key)) edges.set(key, { source, target: END, conditional: false })
    }
    nodes.set(END, { id: END, label: END, kind: 'end' })

    const structureIndex = this.out.length
    this.push(actor.start, 'graph_structure', { agent, nodes: [...nodes.values()], edges: [...edges.values()] })

    // Loose tool / model calls made by the orchestrator itself
    this.items(loose, agent)
    for (const inv of invocations) {
      this.push(inv.start, 'node_start', { agent, node: inv.id, step: inv.step, from: inv.from })
      if (inv.actor) childNames.set(inv, this.actor(inv.actor, agent))
      this.push(inv.end, 'node_end', { agent, node: inv.id, step: inv.step, ...(inv.error ? { error: clip(inv.error, MAX_TASK) } : {}) })
    }
    this.push(actor.end, 'node_start', { agent, node: END, step: Math.max(0, ...invocations.map(i => i.step)) + 1, from: endFrom })

    // Link subgraph nodes to the agent that drew them (names are known now)
    const structure = this.out[structureIndex].payload
    for (const [inv, child] of childNames) {
      const node = (structure.nodes as Record<string, unknown>[]).find(n => n.id === inv.id)
      if (node && node.kind === 'subgraph' && !node.child) node.child = child
    }
  }
}

function spansOfActor(actor: Actor): Span[] {
  const out: Span[] = [actor.span]
  const walk = (items: Item[]) => {
    for (const i of items) {
      if (i.t === 'llm') out.push(i.n.span, ...i.merged)
      else if (i.t === 'tool') walk(i.inner)
      else if (i.t === 'gnode') walk(i.inner)
      else if (i.t === 'actor') { out.push(i.a.span); walk(i.a.items) }
    }
  }
  walk(actor.items)
  return out
}

function taskOf(actor: Actor): string {
  const own = userMessages(actor.span)
  if (own.length) return own[own.length - 1]
  return str(actor.span.attributes['gen_ai.agent.description'])
}

function firstUserMessage(actor: Actor): string {
  for (const span of spansOfActor(actor)) {
    const messages = userMessages(span)
    if (messages.length) return messages[0]
  }
  return ''
}

function summaryOf(actor: Actor): string {
  const own = outputOf(actor.span).text
  if (own.length) return own.join('\n')
  const llms = actor.items.filter((i): i is Extract<Item, { t: 'llm' }> => i.t === 'llm')
  for (let i = llms.length - 1; i >= 0; i--) {
    const text = [llms[i].n.span, ...llms[i].merged].map(outputOf).find(p => p.text.length)?.text
    if (text?.length) return text.join('\n')
  }
  return ''
}

/** Build the span tree, clamping children into their parent's time window so
 *  clock skew between services can't put a child before its parent. */
function tree(spans: Span[]): RawNode[] {
  const ids = new Set(spans.map(s => s.spanId))
  // Measure from the earliest root: a skewed child may claim to start sooner
  const rootStarts = spans.filter(s => !s.parentSpanId || !ids.has(s.parentSpanId)).map(s => s.start)
  const t0 = (rootStarts.length ? rootStarts : spans.map(s => s.start)).reduce((min, t) => (t < min ? t : min))
  const nodes = new Map<string, RawNode>()
  for (const span of spans) {
    const start = Number(span.start - t0)
    nodes.set(span.spanId, { span, start, end: Math.max(start, Number(span.end - t0)), children: [] })
  }
  const roots: RawNode[] = []
  for (const node of nodes.values()) {
    const parent = node.span.parentSpanId ? nodes.get(node.span.parentSpanId) : undefined
    if (parent && parent !== node) {
      node.parent = parent
      parent.children.push(node)
    } else {
      roots.push(node)
    }
  }
  const clamp = (n: RawNode) => {
    n.children.sort((a, b) => a.start - b.start || a.end - b.end)
    for (const c of n.children) {
      c.start = Math.min(Math.max(c.start, n.start), n.end)
      c.end = Math.min(Math.max(c.end, c.start), n.end)
      clamp(c)
    }
  }
  roots.sort((a, b) => a.start - b.start)
  roots.forEach(clamp)
  return roots
}

/** Convert one trace. Returns null when it has nothing to show (e.g. only
 *  a workflow build span). */
export function convertTrace(spans: Span[], options: ConvertOptions = {}): ConvertedTrace | null {
  if (!spans.length) return null
  const roots = tree(spans)
  const top = roots.flatMap(build)
  if (!top.some(i => i.t !== 'gnode' || i.inner.length)) return null

  let main: Actor
  const onlyActor = top.length === 1 && top[0].t === 'actor' ? top[0].a : null
  if (onlyActor) {
    main = onlyActor
  } else {
    const first = roots[0]
    const end = Math.max(...roots.map(r => r.end))
    const service = str(first.span.resource['service.name'])
    const base = roots.length === 1 ? agentName(first.span) : options.fallbackName || service || 'trace'
    main = { span: first.span, start: first.start, end, base, kind: 'agent', items: top }
  }

  const emitter = new Emitter(options.registry)
  const name = emitter.actor(main, null)
  const events = emitter.out
    .sort((a, b) => a.t - b.t || a.seq - b.seq)
    .map(p => ({ time: Math.round(p.t / 1e3) / 1e6, type: p.type, payload: p.payload }))
  return { traceId: spans[0].traceId, name, events, duration: main.end / 1e9 }
}

/** Group spans by trace id, in order of each trace's first span. */
export function groupByTrace(spans: Span[]): Map<string, Span[]> {
  const traces = new Map<string, Span[]>()
  for (const span of spans) {
    const list = traces.get(span.traceId)
    if (list) list.push(span)
    else traces.set(span.traceId, [span])
  }
  return traces
}

/** Convert every trace in a batch of spans (e.g. a whole export file). */
export function convertSpans(spans: Span[], options: ConvertOptions = {}): ConvertedTrace[] {
  const registry = options.registry ?? new Map()
  collectDeclaredGraphs(spans, registry)
  const out: ConvertedTrace[] = []
  for (const traceSpans of groupByTrace(spans).values()) {
    const converted = convertTrace(traceSpans, { ...options, registry })
    if (converted) out.push(converted)
  }
  return out
}

