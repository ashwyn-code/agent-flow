"""Record an OpenTelemetry fixture from one of the adapters' offline demos.

    python scripts/otel/fixtures/record.py <framework>

Run it with a Python environment that has the adapter installed
(`pip install -e adapters/<framework>`) plus `opentelemetry-sdk` and
`opentelemetry-exporter-otlp-proto-common`; for the OpenInference fixtures
also `openinference-instrumentation-langchain` / `-openai-agents`.

The demo runs with the framework's own OpenTelemetry spans (or OpenInference's
instrumentation), exported as OTLP JSON lines. Stack traces, schemas and local
paths are removed and the result is gzipped next to this file.
"""
import argparse
import asyncio
import gzip
import json
import os
import re
import sys
import tempfile

from google.protobuf import json_format
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.join(HERE, "..", "..", "..", "adapters")
FIXTURE = {
    "strands": "strands",
    "agent-framework": "agent-framework",
    "google-adk": "google-adk",
    "openai-agents": "openai-agents-openinference",
    "langgraph": "langgraph-openinference",
}
DROP = {"exception.stacktrace", "gen_ai.tool.definitions", "gen_ai.tool.json_schema",
        "llm.invocation_parameters", "tool.parameters", "gen_ai.system_instructions"}


class Collect(SpanExporter):
    def __init__(self):
        self.lines = []

    def export(self, spans):
        self.lines.append(json_format.MessageToJson(encode_spans(spans), indent=None))
        return SpanExportResult.SUCCESS


def scrub(text):
    return re.sub(r"/(private/)?(tmp|var|Users|home)/[^\s\"'\\]*", "/path", text)


def prune(value):
    if isinstance(value, dict):
        return {k: prune(v) for k, v in value.items() if k not in ("tools", "system_instruction", "tool_config", "labels")}
    if isinstance(value, list):
        return [prune(v) for v in value]
    return value


def clean(attrs):
    out = []
    for kv in attrs:
        if kv["key"] in DROP:
            continue
        value = kv.get("value", {})
        if "stringValue" in value:
            text = value["stringValue"]
            if kv["key"] in ("gcp.vertex.agent.llm_request", "gcp.vertex.agent.llm_response", "metadata"):
                try:
                    text = json.dumps(prune(json.loads(text)))
                except ValueError:
                    pass
            value = {"stringValue": scrub(text)}
        out.append({"key": kv["key"], "value": value})
    return out


def run_demo(framework, provider, out):
    sys.path.insert(0, os.path.join(ADAPTERS, framework, "examples"))
    sys.path.insert(0, os.path.join(ADAPTERS, framework))
    if framework == "langgraph":
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=provider)
        import deep_orchestration_demo as demo
        demo.run(out, 0.0)
    elif framework == "openai-agents":
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
        import agent_flow_openai_agents
        import support_desk_demo as demo
        OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
        install = agent_flow_openai_agents.install
        demo.install = lambda *a, **k: install(*a, **{**k, "exclusive": False})  # keep OpenInference's processor
        asyncio.run(demo.run(out, 0.0))
    elif framework == "strands":
        import orchestration_demo as demo
        demo.run(out, 0.0)
    elif framework == "agent-framework":
        from agent_framework.observability import enable_instrumentation
        enable_instrumentation(enable_sensitive_data=True)
        import incident_response_demo as demo
        asyncio.run(demo.run(out, 0.0))
    elif framework == "google-adk":
        import trip_planner_demo as demo
        asyncio.run(demo.run(out, 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=sorted(FIXTURE))
    args = parser.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "x")
    os.environ.setdefault("GOOGLE_API_KEY", "x")
    collect = Collect()
    provider = TracerProvider(resource=Resource.create({"service.name": f"{args.framework}-demo"}))
    provider.add_span_processor(SimpleSpanProcessor(collect))
    trace.set_tracer_provider(provider)
    run_demo(args.framework, provider, os.path.join(tempfile.mkdtemp(), "events.jsonl"))
    provider.shutdown()

    lines = []
    for line in collect.lines:
        request = json.loads(line)
        for rs in request["resourceSpans"]:
            rs["resource"]["attributes"] = clean(rs["resource"].get("attributes", []))
            for ss in rs["scopeSpans"]:
                for span in ss["spans"]:
                    span["attributes"] = clean(span.get("attributes", []))
                    for event in span.get("events", []):
                        event["attributes"] = clean(event.get("attributes", []))
                    if "message" in span.get("status", {}):
                        span["status"]["message"] = scrub(span["status"]["message"])
        lines.append(json.dumps(request, separators=(",", ":")))
    path = os.path.join(HERE, f"{FIXTURE[args.framework]}.otlp.jsonl.gz")
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(("\n".join(lines) + "\n").encode())
    print("wrote", path)
    if args.framework == "google-adk":
        # The protobuf fixture is the same spans, for the decoder's equivalence test
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        merged = ExportTraceServiceRequest()
        for line in lines:
            request = ExportTraceServiceRequest()
            json_format.Parse(line, request)
            merged.resource_spans.extend(request.resource_spans)
        pb = os.path.join(HERE, "google-adk.otlp.pb.gz")
        with gzip.open(pb, "wb", compresslevel=9) as f:
            f.write(merged.SerializeToString())
        print("wrote", pb)


if __name__ == "__main__":
    main()
