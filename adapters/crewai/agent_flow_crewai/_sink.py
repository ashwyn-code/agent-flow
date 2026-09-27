# Vendored copy of adapters/_shared/agent_flow_sink.py: edit that file and run adapters/sync_sink.py
"""Where Agent Flow adapters send their events.

This file is vendored into every adapter as ``_sink.py`` (see
``adapters/sync_sink.py``); edit it here and re-run the sync script.

An :class:`EventSink` takes events from the adapter's callbacks and delivers
them from a background thread, so the agent never waits on disk or network:

- **Transport:** a local JSONL file (``path`` / ``AGENT_FLOW_EVENT_LOG``) or
  an Agent Flow relay's ``/ingest`` endpoint (``url`` / ``AGENT_FLOW_URL``,
  with ``token`` / ``AGENT_FLOW_TOKEN``).
- **Never blocks:** events go into a bounded queue; when it is full, new
  events are dropped and counted. Failed HTTP batches are retried briefly,
  then dropped. Nothing here raises into the agent.
- **Content control:** ``content="metadata"`` (``AGENT_FLOW_CONTENT``) keeps
  the shape of a run (agents, tools, timings, errors, tokens) but replaces
  prompts, arguments, results and messages with their length. A ``redact``
  callable ``(event_type, payload) -> payload | None`` can rewrite or drop
  any event on top of that.
- **Sampling:** ``sample_rate`` (``AGENT_FLOW_SAMPLE_RATE``) keeps that
  fraction of top-level runs, deciding once per run so a kept run is complete.
"""

import atexit
import json
import logging
import os
import queue
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
import weakref
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("agent_flow")

Redactor = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]

_LIVE: "weakref.WeakSet[EventSink]" = weakref.WeakSet()


def flush_all(timeout: float = 5.0) -> None:
    """Flush every sink in this process (e.g. before reading their files)."""
    for sink in list(_LIVE):
        sink.flush(timeout)

_CONTENT_FIELDS = {
    "message": ("content",),
    "agent_spawn": ("task",),
    "subagent_dispatch": ("task",),
    "subagent_return": ("summary",),
    "tool_call_start": ("args",),
    "tool_call_end": ("result", "errorMessage"),
    "node_end": ("error",),
}


def _placeholder(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return f"[{len(text)} chars]" if text else ""


def metadata_only(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep a run's shape; replace user and model content with its length."""
    fields = _CONTENT_FIELDS.get(event_type)
    if not fields and "inputData" not in payload:
        return payload
    out = dict(payload)
    for field in fields or ():
        if out.get(field):
            out[field] = _placeholder(out[field])
    out.pop("inputData", None)
    return out


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("agent-flow: ignoring %s=%r (not a number)", name, raw)
        return None


class EventSink:
    """Deliver Agent Flow events to a file or a relay without blocking the agent.

    Args:
        path: JSONL file (defaults to ``$AGENT_FLOW_EVENT_LOG``, then
            ``agent-flow-events.jsonl``). Ignored when ``url`` is set.
        truncate: Empty the file before the first event is written.
        url: A relay's ingest endpoint, e.g. ``http://agent-flow:3001/ingest``
            (defaults to ``$AGENT_FLOW_URL``).
        token: Bearer token for the relay (defaults to ``$AGENT_FLOW_TOKEN``).
        session: Label for this sender's session tab in the relay.
        content: ``"full"`` (default) or ``"metadata"`` (defaults to
            ``$AGENT_FLOW_CONTENT``).
        redact: ``(event_type, payload) -> payload | None``; ``None`` drops it.
        sample_rate: Fraction of top-level runs to keep, 0–1 (defaults to
            ``$AGENT_FLOW_SAMPLE_RATE``, then 1).
        max_queue: Events held in memory before new ones are dropped.
        flush_interval: Seconds between background flushes.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        truncate: bool = False,
        url: Optional[str] = None,
        token: Optional[str] = None,
        session: Optional[str] = None,
        content: Optional[str] = None,
        redact: Optional[Redactor] = None,
        sample_rate: Optional[float] = None,
        max_queue: int = 10_000,
        flush_interval: float = 0.2,
    ) -> None:
        self.url = url or os.environ.get("AGENT_FLOW_URL") or None
        self.path = path or os.environ.get("AGENT_FLOW_EVENT_LOG") or "agent-flow-events.jsonl"
        self.token = token or os.environ.get("AGENT_FLOW_TOKEN") or None
        self.session_id = uuid.uuid4().hex
        self.session_label = session
        self.content = (content or os.environ.get("AGENT_FLOW_CONTENT") or "full").lower()
        self.redact = redact
        rate = sample_rate if sample_rate is not None else _env_float("AGENT_FLOW_SAMPLE_RATE")
        self.sample_rate = 1.0 if rate is None else max(0.0, min(1.0, rate))
        self.flush_interval = flush_interval
        self.dropped = 0  # events lost to a full queue or a failed delivery
        self._truncate = truncate
        self._start: Optional[float] = None
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._warned: Set[str] = set()
        self._closed = False
        # Sampling: agent -> parent, and the decision per top-level run
        self._parents: Dict[str, Optional[str]] = {}
        self._kept_roots: Set[str] = set()
        self._dropped_roots: Set[str] = set()
        self._idle = threading.Event()
        self._idle.set()
        self._worker = threading.Thread(target=self._run, name="agent-flow-sink", daemon=True)
        self._worker.start()
        _LIVE.add(self)
        atexit.register(self.close)

    # ─── Called from adapter callbacks ───────────────────────────────────────

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Queue one event. Never blocks and never raises."""
        try:
            now = time.monotonic()
            with self._lock:
                if self._closed:
                    return
                if self._start is None:
                    self._start = now
                if not self._sampled(event_type, payload):
                    return
                if self.content == "metadata":
                    payload = metadata_only(event_type, payload)
                if self.redact is not None:
                    result = self.redact(event_type, payload)
                    if result is None:
                        return
                    payload = result
                event = {"time": round(now - self._start, 3), "type": event_type, "payload": payload}
            self._idle.clear()
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
            self._warn_once("full", "agent-flow: event queue full; dropping events (the visualizer is falling behind)")
        except Exception:
            self._warn_once("emit", "agent-flow: failed to queue an event; further errors are suppressed", exc=True)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until everything queued so far has been delivered (or dropped)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.empty() and self._idle.is_set():
                return
            time.sleep(0.01)

    def close(self, timeout: float = 5.0) -> None:
        """Flush and stop the background thread. Safe to call more than once."""
        if self._closed:
            return
        self.flush(timeout)
        with self._lock:
            self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout)

    # ─── Sampling ────────────────────────────────────────────────────────────

    def _root(self, name: Optional[str]) -> Optional[str]:
        seen = 0
        while name is not None and seen < 100:
            parent = self._parents.get(name)
            if parent is None:
                return name
            name, seen = parent, seen + 1
        return name

    def _sampled(self, event_type: str, payload: Dict[str, Any]) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if event_type == "agent_spawn":
            name = payload.get("name")
            if payload.get("isMain") or not payload.get("parent"):
                self._parents[name] = None
                if random.random() < self.sample_rate:
                    self._kept_roots.add(name)
                    self._dropped_roots.discard(name)
                else:
                    self._dropped_roots.add(name)
                    self._kept_roots.discard(name)
            else:
                self._parents[name] = payload.get("parent")
        agent = (payload.get("name") or payload.get("agent") or payload.get("parent")
                 if event_type != "subagent_dispatch" else payload.get("parent"))
        return self._root(agent) in self._kept_roots

    # ─── Background delivery ─────────────────────────────────────────────────

    def _run(self) -> None:
        stop = False
        while not stop:
            batch: List[Dict[str, Any]] = []
            try:
                item = self._queue.get(timeout=self.flush_interval)
                if item is None:
                    stop = True
                else:
                    batch.append(item)
                while len(batch) < 500:
                    item = self._queue.get_nowait()
                    if item is None:
                        stop = True
                        break
                    batch.append(item)
            except queue.Empty:
                pass
            if batch:
                try:
                    self._deliver(batch)
                except Exception:
                    self.dropped += len(batch)
                    self._warn_once("deliver", "agent-flow: failed to deliver events; further errors are suppressed", exc=True)
            if self._queue.empty():
                self._idle.set()

    def _deliver(self, batch: List[Dict[str, Any]]) -> None:
        if self.url:
            self._post(batch)
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        mode = "w" if self._truncate else "a"
        self._truncate = False
        with open(self.path, mode, encoding="utf-8") as f:
            f.write("".join(json.dumps(e, default=str, ensure_ascii=False) + "\n" for e in batch))

    def _post(self, batch: List[Dict[str, Any]]) -> None:
        body = json.dumps({"session": {"id": self.session_id, "label": self.session_label}, "events": batch},
                          default=str, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in range(3):
            try:
                request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=3) as response:
                    response.read()
                return
            except urllib.error.HTTPError as err:
                if err.code in (400, 401, 403, 413):  # not worth retrying
                    self.dropped += len(batch)
                    self._warn_once(f"http{err.code}", f"agent-flow: relay rejected events (HTTP {err.code}); check the URL and token")
                    return
            except Exception:
                pass
            time.sleep(0.2 * (attempt + 1))
        self.dropped += len(batch)
        self._warn_once("unreachable", f"agent-flow: relay at {self.url} is unreachable; dropping events")

    def _warn_once(self, key: str, message: str, exc: bool = False) -> None:
        if key not in self._warned:
            self._warned.add(key)
            (logger.exception if exc else logger.warning)(message)
