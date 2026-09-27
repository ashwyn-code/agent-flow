"""Vendor adapters/_shared/agent_flow_sink.py into every adapter package as _sink.py.

Each adapter is installed on its own, so they carry a copy instead of
depending on a shared package. Run after editing the shared file:

    python adapters/sync_sink.py          # write the copies
    python adapters/sync_sink.py --check  # exit 1 if any copy is stale
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "_shared" / "agent_flow_sink.py"
PACKAGES = {
    "langgraph": "agent_flow_langgraph",
    "strands": "agent_flow_strands",
    "agent-framework": "agent_flow_agent_framework",
    "crewai": "agent_flow_crewai",
    "openai-agents": "agent_flow_openai_agents",
    "google-adk": "agent_flow_adk",
}
HEADER = "# Vendored copy of adapters/_shared/agent_flow_sink.py: edit that file and run adapters/sync_sink.py\n"


def main() -> int:
    text = HEADER + SOURCE.read_text()
    stale = []
    for folder, package in PACKAGES.items():
        target = HERE / folder / package / "_sink.py"
        if "--check" in sys.argv:
            if not target.exists() or target.read_text() != text:
                stale.append(str(target.relative_to(HERE.parent)))
        else:
            target.write_text(text)
            print("wrote", target.relative_to(HERE.parent))
    if stale:
        print("stale sink copies (run python adapters/sync_sink.py):", *stale, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
