"""Keep CrewAI offline and out of the real home directory during tests."""
import os
import tempfile

_home = tempfile.mkdtemp(prefix="agent-flow-crewai-")
os.environ.update({
    "HOME": _home,
    "CREWAI_STORAGE_DIR": os.path.join(_home, "storage"),
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "CREWAI_TRACING_ENABLED": "false",
})
