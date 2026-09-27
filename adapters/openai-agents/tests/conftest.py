"""No network: a dummy key, and every test installs its processor exclusively."""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
