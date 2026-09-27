# OpenTelemetry fixtures

Traces recorded from the adapters' offline demos with OpenTelemetry tracing on, used by `convert.test.ts`, `otlp.test.ts` and `receiver.test.ts`:

| File | Spans from |
|---|---|
| `strands.otlp.jsonl.gz` | Strands Agents' own tracing (GenAI conventions) |
| `agent-framework.otlp.jsonl.gz` | Microsoft Agent Framework's own tracing, including the `workflow.build` spans |
| `google-adk.otlp.jsonl.gz` | Google ADK's own tracing |
| `google-adk.otlp.pb.gz` | The same spans as one protobuf `ExportTraceServiceRequest` (written with the JSON one) |
| `openai-agents-openinference.otlp.jsonl.gz` | OpenInference's OpenAI Agents instrumentation |
| `langgraph-openinference.otlp.jsonl.gz` | OpenInference's LangChain instrumentation on the deep orchestration demo |

Each is OTLP JSON, one export request per line, gzipped. Re-record one with `python scripts/otel/fixtures/record.py <framework>` (see its docstring for what to install).
