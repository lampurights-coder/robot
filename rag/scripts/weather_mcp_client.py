#!/usr/bin/env python3
"""Local weather tool from tita-arshia/agent (no remote MCP, no email).

Uses weather_mcp.service.get_weather(city) in-process.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(
    os.environ.get(
        "WEATHER_AGENT_ROOT",
        Path(__file__).resolve().parents[2] / "agent",
    )
).resolve()

if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

# Load agent .env (LANGSEARCH / OpenAI for the weather tool only)
try:
    from dotenv import load_dotenv

    load_dotenv(AGENT_ROOT / ".env", override=False)
except Exception:
    pass

_WEATHER_RE = re.compile(
    r"(آب\s*و\s*هوا|هواشناسی|هوای\s+\w+|دما|درجه\s*حرارت|"
    r"باران|برف|آفتابی|cloudy|weather|temperature|forecast|"
    r"چطوره\s*هوا|هوا\s*چطور)",
    re.I,
)
_JUNK_LOC = re.compile(
    r"^(چطوره?|چطور|چیه|چیست|است|هست|today|now|please)$", re.I
)


def is_weather_question(text: str) -> bool:
    return bool(_WEATHER_RE.search(text or ""))


def guess_location(text: str, default: str = "Tehran") -> str:
    m = re.search(r"هوای?\s+([A-Za-z\u0600-\u06FF]+)", text or "")
    if m:
        city = m.group(1)
        if city and not _JUNK_LOC.match(city):
            return city
    m2 = re.search(
        r"(?:weather|forecast)\s+(?:in|for)\s+([A-Za-z][\w\s\-]{1,40})",
        text or "",
        re.I,
    )
    if m2:
        return m2.group(1).strip().split()[0]
    return default


def call_weather(location: str, **kwargs: Any) -> dict[str, Any]:
    """Call local get_weather tool; returns dict for the FAQ API."""
    from weather_mcp.service import get_weather

    result = get_weather(location, **kwargs)
    data = result.model_dump(mode="json")
    answer = (result.answer_fa or "").strip()
    if result.temp_high_c is not None or result.temp_low_c is not None:
        answer += (
            f"\n(دمای تقریبی: {result.temp_low_c}–{result.temp_high_c} °C)"
            if result.temp_low_c is not None and result.temp_high_c is not None
            else ""
        )
    return {
        "tool": "get_weather",
        "args": {"city": location},
        "text": answer or str(data),
        "raw": data,
        "url": "local:weather_mcp.service.get_weather",
    }


if __name__ == "__main__":
    import json

    loc = sys.argv[1] if len(sys.argv) > 1 else "Tehran"
    print(json.dumps(call_weather(loc), ensure_ascii=False, indent=2))
