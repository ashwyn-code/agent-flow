"""Rule-based fake CrewAI LLM for demos and tests (no API key, no network).

CrewAI agents drive tools through ReAct text ("Thought / Action / Action
Input" then "Final Answer"), so the fake returns that text. The reply is a
pure function of the conversation so far, so one instance is safe to share.
Custom LLMs must emit CrewAI's LLM call events themselves; this one does.
"""

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context

ToolCall = Tuple[str, Dict[str, Any]]
# policy(messages) -> completion text
Policy = Callable[[List[Dict[str, Any]]], str]

_OBSERVATION = re.compile(r"\nObservation:\s*(.*?)(?=\n(?:Thought|Action|Final Answer)\b|\Z)", re.DOTALL)


class RuleBasedLLM(BaseLLM):
    policy: Any = None
    delay: Any = None  # Callable[[], float]

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None, available_functions: Any = None,
             from_task: Any = None, from_agent: Any = None, response_model: Any = None, **kwargs: Any) -> str:
        messages = messages if isinstance(messages, list) else [{"role": "user", "content": str(messages)}]
        with llm_call_context():
            self._emit_call_started_event(messages=messages, tools=tools, from_task=from_task, from_agent=from_agent)
            if self.delay is not None:
                time.sleep(self.delay())
            text = self.policy(messages)
            prompt = sum(len(str(m.get("content", ""))) for m in messages) // 4
            self._emit_call_completed_event(
                response=text, call_type=LLMCallType.LLM_CALL, from_task=from_task, from_agent=from_agent,
                messages=messages, usage={"prompt_tokens": prompt, "completion_tokens": 60, "total_tokens": prompt + 60})
            return text


def fake_llm(policy: Policy, model: str = "demo-gpt-4o-mini", delay: Optional[Callable[[], float]] = None) -> RuleBasedLLM:
    return RuleBasedLLM(model=model, policy=policy, delay=delay)


def task_text(messages: Sequence[Dict[str, Any]]) -> str:
    """The task the agent was given (CrewAI's "Current Task:" prompt, or the first user message)."""
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "user":
            match = re.search(r"Current Task:\s*(.*?)(?:\n\n|\Z)", content, re.DOTALL)
            return match.group(1).strip() if match else content.strip()
    return ""


def observations(messages: Sequence[Dict[str, Any]]) -> List[str]:
    """Tool results so far, in order."""
    found: List[str] = []
    for message in messages:
        if message.get("role") == "assistant":
            found += [m.strip() for m in _OBSERVATION.findall(str(message.get("content", "")))]
    return found


def _failed(observation: str) -> bool:
    lowered = observation.lower()
    return lowered.startswith("error") or "error executing tool" in lowered or "i encountered an error" in lowered


def react(plan: Callable[[str], List[ToolCall]], final: Callable[[str], str],
          thinking: Optional[Callable[[str], str]] = None) -> Policy:
    """ReAct policy: call ``plan(task)`` tools one per turn (a failed call is
    retried once), then give the final answer."""

    def policy(messages: List[Dict[str, Any]]) -> str:
        task = task_text(messages)
        steps = plan(task)
        results = observations(messages)
        done = sum(1 for r in results if not _failed(r))
        failures = len(results) - done
        # Give up on a step after it fails twice
        done += failures // 2
        if done < len(steps):
            name, arguments = steps[done]
            thought = (thinking(task) if thinking and not results else f"Next: {name}")
            return f"Thought: {thought}\nAction: {name}\nAction Input: {json.dumps(arguments)}"
        return f"Thought: I have what I need.\nFinal Answer: {final(task)}"

    return policy


def answer(text: str, thought: str = "I know this.") -> Policy:
    return lambda messages: f"Thought: {thought}\nFinal Answer: {text}"
