"""Nested, fan-out Google ADK demo for Agent Flow. No API key, no network.

    trip_planner (SequentialAgent)
    ├─ intake (LlmAgent)
    ├─ research (ParallelAgent)
    │    flights (a search fails once and is retried) │ hotels (2 parallel calls) │ weather
    ├─ itinerary_loop (LoopAgent, max 3)
    │    planner ─ asks budget_analyst (AgentTool) ─ currency_convert
    │    critic ── revises once, then calls exit_loop
    └─ booking (LlmAgent) ──transfer_to_agent──▶ reservations (sub-agent)

Every ADK agent pattern in one run: Sequential, Parallel and Loop workflow
agents (nested), parallel tool calls, an agent used as a tool, LLM-driven
transfer to a sub-agent, and a tool that reports an error, four levels deep.

    python examples/trip_planner_demo.py --out agent-flow.jsonl
"""

import argparse
import asyncio
import logging
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import exit_loop
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_llm import fake_llm, react  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_flow_adk import AgentFlowPlugin  # noqa: E402

DELAY = 0.5
REQUEST = "Plan a 4-day trip to Lisbon in October for two, mid-range budget, and book it"


async def pause(key: str) -> None:
    if DELAY > 0:
        await asyncio.sleep(DELAY * (0.6 + (zlib.crc32(key.encode()) % 9) / 10))


def llm(policy: Any, model: str = "demo-gemini-2.5-flash") -> Any:
    return fake_llm(policy, model=model, delay=lambda: DELAY * 0.4)


def say(text: str, thinking: Optional[str] = None) -> Any:
    return lambda contents: {"text": text, **({"thinking": thinking} if thinking else {})}


# ─── Tools ───────────────────────────────────────────────────────────────────

_flight_searches: List[int] = []


async def lookup_profile(traveler: str) -> dict:
    """Look up a traveler's preferences."""
    await pause("profile")
    return {"status": "ok", "seat": "aisle", "diet": "vegetarian", "loyalty": "TAP Miles&Go"}


async def search_flights(origin: str, destination: str) -> dict:
    """Search flights."""
    await pause("flights")
    _flight_searches.append(1)
    if len(_flight_searches) == 1:
        return {"status": "error", "error_message": "fares API timed out"}
    return {"status": "ok", "best": "TP1331 LHR→LIS 08:40, €142 return"}


async def search_hotels(area: str) -> dict:
    """Search hotels in an area."""
    await pause("hotels:" + area)
    return {"status": "ok", "area": area, "pick": f"4★ boutique in {area}, €118/night"}


async def get_forecast(city: str) -> dict:
    """Get a weather forecast."""
    await pause("weather")
    return {"status": "ok", "forecast": "22°C, mostly sunny, one rainy afternoon"}


async def currency_convert(amount_eur: float, currency: str) -> dict:
    """Convert euros to another currency."""
    await pause("fx")
    return {"status": "ok", "amount": round(amount_eur * 0.85, 2), "currency": currency}


async def draft_itinerary(days: int) -> dict:
    """Draft a day-by-day itinerary."""
    await pause("draft")
    return {"status": "ok", "days": days, "outline": "Alfama & castle, Belém, Sintra day trip, LX Factory"}


async def reserve(item: str) -> dict:
    """Reserve a flight or hotel."""
    await pause("reserve:" + item)
    return {"status": "ok", "confirmation": f"{item.upper()[:3]}-{zlib.crc32(item.encode()) % 90000 + 10000}"}


# ─── Agents ──────────────────────────────────────────────────────────────────

def build() -> SequentialAgent:
    intake = LlmAgent(name="intake", description="Understands the request", tools=[lookup_profile], model=llm(react(
        lambda task: [[("lookup_profile", {"traveler": "primary"})]],
        lambda task, r: "Two travellers, 4 days in Lisbon in October, mid-range, vegetarian, aisle seats.",
        thinking=lambda task: "Load the traveller profile first.",
    )))

    flights = LlmAgent(name="flights", tools=[search_flights], model=llm(react(
        lambda task: [[("search_flights", {"origin": "LHR", "destination": "LIS"})]],
        lambda task, r: "Best fare: TP1331, €142 return.")))
    hotels = LlmAgent(name="hotels", tools=[search_hotels], model=llm(react(
        lambda task: [[("search_hotels", {"area": "Chiado"}), ("search_hotels", {"area": "Alfama"})]],
        lambda task, r: "Shortlist: boutique hotels in Chiado and Alfama, about €118/night.",
        thinking=lambda task: "Compare two neighbourhoods at once.")))
    weather = LlmAgent(name="weather", tools=[get_forecast], model=llm(react(
        lambda task: [[("get_forecast", {"city": "Lisbon"})]],
        lambda task, r: "22°C and mostly sunny; plan Sintra for a dry day.")))
    research = ParallelAgent(name="research", sub_agents=[flights, hotels, weather])

    budget_analyst = LlmAgent(name="budget_analyst", description="Checks costs", tools=[currency_convert], model=llm(react(
        lambda task: [[("currency_convert", {"amount_eur": 1320, "currency": "GBP"})]],
        lambda task, r: "Total about €1,320 (£1,122): within the mid-range budget.")))
    planner = LlmAgent(name="planner", tools=[draft_itinerary, AgentTool(agent=budget_analyst)], model=llm(react(
        lambda task: [[("draft_itinerary", {"days": 4}), ("budget_analyst", {"request": "Cost this 4-day Lisbon plan"})]],
        lambda task, r: "Itinerary v2: Alfama, Belém, Sintra (dry day), LX Factory; €1,320 total.",
        thinking=lambda task: "Draft the days and check the budget in parallel.",
    ), model="demo-gemini-2.5-pro"))
    reviews: List[int] = []

    def critic_policy(contents: Any) -> Dict[str, Any]:
        reviews.append(1)
        if len(reviews) == 1:
            return {"text": "Revise: Sintra needs a full day and a dry forecast.", "thinking": "Day 3 is overpacked."}
        return {"tools": [("exit_loop", {})], "thinking": "This version works."}

    critic = LlmAgent(name="critic", tools=[exit_loop], model=llm(critic_policy, model="demo-gemini-2.5-pro"))
    itinerary_loop = LoopAgent(name="itinerary_loop", sub_agents=[planner, critic], max_iterations=3)

    reservations = LlmAgent(name="reservations", description="Makes reservations", tools=[reserve], model=llm(react(
        lambda task: [[("reserve", {"item": "flight TP1331"}), ("reserve", {"item": "hotel Chiado"})]],
        lambda task, r: "Booked: flight and hotel confirmed.")))
    booking = LlmAgent(name="booking", description="Coordinates booking", sub_agents=[reservations], model=llm(react(
        lambda task: [[("transfer_to_agent", {"agent_name": "reservations"})]],
        lambda task, r: "Handing over to reservations.")))

    return SequentialAgent(name="trip_planner", sub_agents=[intake, research, itinerary_loop, booking])


async def run(out: str, delay: float = DELAY) -> str:
    global DELAY
    DELAY = delay
    _flight_searches.clear()
    plugin = AgentFlowPlugin(out, truncate=True)
    app = App(name="trip_planner", root_agent=build(), plugins=[plugin])
    runner = Runner(app=app, session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name="trip_planner", user_id="demo")
    final = ""
    async for event in runner.run_async(user_id="demo", session_id=session.id,
                                        new_message=types.Content(role="user", parts=[types.Part(text=REQUEST)])):
        for part in (event.content.parts if event.content else None) or []:
            if part.text and not part.thought:
                final = part.text
    return final


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="agent-flow-events.jsonl", help="JSONL file to write")
    parser.add_argument("--delay", type=float, default=DELAY, help="base seconds per tool / model call")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    print("final:", asyncio.run(run(args.out, args.delay)))
    print(f"events written to {args.out}")


if __name__ == "__main__":
    main()
