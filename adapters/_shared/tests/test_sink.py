"""Tests for the shared event sink (adapters/_shared/agent_flow_sink.py)."""
import http.server
import json
import os
import random
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_flow_sink import EventSink, flush_all, metadata_only  # noqa: E402


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_events(sink, run="main", child="child"):
    sink.emit("agent_spawn", {"name": run, "isMain": True, "task": "secret prompt"})
    sink.emit("message", {"agent": run, "role": "user", "content": "my card is 4242"})
    sink.emit("subagent_dispatch", {"parent": run, "child": child, "task": "look it up"})
    sink.emit("agent_spawn", {"name": child, "parent": run, "task": "look it up"})
    sink.emit("tool_call_start", {"agent": child, "tool": "lookup", "args": '{"q": "x"}', "inputData": {"q": "x"}})
    sink.emit("tool_call_end", {"agent": child, "tool": "lookup", "result": "found x", "isError": True, "errorMessage": "boom"})
    sink.emit("agent_complete", {"name": child})
    sink.emit("agent_complete", {"name": run})


def test_file_delivery_in_order(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text("stale\n")
    sink = EventSink(str(path), truncate=True)
    for i in range(1200):  # more than one batch
        sink.emit("context_update", {"agent": "a", "tokens": i})
    sink.close()
    events = lines(path)
    assert [e["payload"]["tokens"] for e in events] == list(range(1200))
    times = [e["time"] for e in events]
    assert times == sorted(times)


def test_emit_never_blocks_and_drops_when_full(tmp_path, monkeypatch):
    sink = EventSink(str(tmp_path / "e.jsonl"), max_queue=10)
    gate = threading.Event()
    monkeypatch.setattr(sink, "_deliver", lambda batch: gate.wait(5))  # a stalled disk / network
    started = time.monotonic()
    for i in range(100):
        sink.emit("context_update", {"agent": "a", "tokens": i})
    assert time.monotonic() - started < 0.5   # never waited on delivery
    assert sink.dropped >= 80
    gate.set()
    sink.close()


def test_a_failing_transport_never_raises(tmp_path, monkeypatch):
    sink = EventSink(str(tmp_path / "e.jsonl"))
    monkeypatch.setattr(sink, "_deliver", lambda batch: (_ for _ in ()).throw(OSError("disk full")))
    sink.emit("message", {"agent": "a", "content": "x"})
    sink.flush()
    assert sink.dropped == 1
    sink.close()


def test_metadata_only_keeps_the_shape(tmp_path):
    path = tmp_path / "e.jsonl"
    sink = EventSink(str(path), content="metadata")
    run_events(sink)
    sink.close()
    by_type = {}
    for e in lines(path):
        by_type.setdefault(e["type"], []).append(e["payload"])
    assert by_type["agent_spawn"][0] == {"name": "main", "isMain": True, "task": "[13 chars]"}
    assert by_type["message"][0]["content"] == "[15 chars]"
    start = by_type["tool_call_start"][0]
    assert start["tool"] == "lookup" and start["args"] == "[10 chars]" and "inputData" not in start
    end = by_type["tool_call_end"][0]
    assert end["isError"] is True and end["result"] == "[7 chars]" and end["errorMessage"] == "[4 chars]"
    assert "4242" not in path.read_text() and "secret" not in path.read_text()


def test_metadata_only_function():
    assert metadata_only("context_update", {"agent": "a", "tokens": 5}) == {"agent": "a", "tokens": 5}
    assert metadata_only("message", {"agent": "a", "content": ""}) == {"agent": "a", "content": ""}


def test_redact_can_rewrite_or_drop(tmp_path):
    path = tmp_path / "e.jsonl"

    def redact(event_type, payload):
        if event_type == "message":
            return None  # drop every message
        if event_type == "tool_call_start":
            return {**payload, "args": "***", "inputData": {}}
        return payload

    sink = EventSink(str(path), redact=redact)
    run_events(sink)
    sink.close()
    events = lines(path)
    assert "message" not in {e["type"] for e in events}
    assert next(e["payload"] for e in events if e["type"] == "tool_call_start")["args"] == "***"


@pytest.mark.parametrize("rate,expected", [(0.0, 0), (1.0, 16)])
def test_sampling_extremes(tmp_path, rate, expected):
    path = tmp_path / "e.jsonl"
    sink = EventSink(str(path), sample_rate=rate)
    run_events(sink, "run1", "child1")
    run_events(sink, "run2", "child2")
    sink.close()
    count = len(lines(path)) if path.exists() else 0
    assert count == expected


def test_sampling_keeps_or_drops_whole_runs(tmp_path, monkeypatch):
    path = tmp_path / "e.jsonl"
    rolls = iter([0.1, 0.9, 0.2])  # keep run1, drop run2, keep run3 at rate 0.5
    monkeypatch.setattr(random, "random", lambda: next(rolls))
    sink = EventSink(str(path), sample_rate=0.5)
    for n in (1, 2, 3):
        run_events(sink, f"run{n}", f"child{n}")
    sink.close()
    names = {e["payload"].get("name") or e["payload"].get("agent") or e["payload"].get("parent") for e in lines(path)}
    assert names == {"run1", "child1", "run3", "child3"}   # every event of a kept run, none of a dropped one
    assert len(lines(path)) == 16


def test_env_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FLOW_EVENT_LOG", str(tmp_path / "env.jsonl"))
    monkeypatch.setenv("AGENT_FLOW_CONTENT", "metadata")
    monkeypatch.setenv("AGENT_FLOW_SAMPLE_RATE", "0.25")
    sink = EventSink()
    assert sink.path.endswith("env.jsonl") and sink.content == "metadata" and sink.sample_rate == 0.25
    sink.close()


def test_flush_all(tmp_path):
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    sinks = [EventSink(str(p)) for p in paths]
    for sink in sinks:
        sink.emit("message", {"agent": "a", "content": "x"})
    flush_all()
    assert all(len(lines(p)) == 1 for p in paths)
    for sink in sinks:
        sink.close()


# ─── HTTP ────────────────────────────────────────────────────────────────────

class _Relay(http.server.BaseHTTPRequestHandler):
    status = 200
    received = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).received.append((self.headers.get("Authorization"), json.loads(body)))
        self.send_response(type(self).status)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def relay():
    handler = type("Relay", (_Relay,), {"received": [], "status": 200})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield handler, f"http://127.0.0.1:{server.server_address[1]}/ingest"
    server.shutdown()


def test_http_delivery(relay):
    handler, url = relay
    sink = EventSink(url=url, token="s3cret", session="checkout-service")
    run_events(sink)
    sink.close()
    auths = {auth for auth, _ in handler.received}
    assert auths == {"Bearer s3cret"}
    bodies = [body for _, body in handler.received]
    assert {b["session"]["id"] for b in bodies} == {sink.session_id}
    assert bodies[0]["session"]["label"] == "checkout-service"
    events = [e for b in bodies for e in b["events"]]
    assert [e["type"] for e in events][:2] == ["agent_spawn", "message"] and len(events) == 8


def test_http_rejection_is_not_retried(relay):
    handler, url = relay
    handler.status = 401
    sink = EventSink(url=url, token="wrong")
    sink.emit("message", {"agent": "a", "content": "x"})
    sink.close()
    assert len(handler.received) == 1 and sink.dropped == 1


def test_unreachable_relay_drops_without_raising():
    sink = EventSink(url="http://127.0.0.1:9/ingest")  # discard port: nothing listens
    started = time.monotonic()
    sink.emit("message", {"agent": "a", "content": "x"})
    assert time.monotonic() - started < 0.1
    sink.close(timeout=10)
    assert sink.dropped == 1
