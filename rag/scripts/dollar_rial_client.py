#!/usr/bin/env python3
"""Market prices (dollar/gold/coin/…) from ecotrust.ir scraper — no web search."""

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

_MARKET_RE = re.compile(
    r"(قیمت\s*دلار|نرخ\s*دلار|دلار\s*(آمریکا|امروز|چند|چقدر|مجازی)?|"
    r"تتر|usdt|usd\b|dollar|"
    r"قیمت\s*طلا|طلای\s*۱۸|طلای\s*18|اونس\s*طلا|قیمت\s*سکه|سکه\s*(امامی|بهار)|"
    r"نیم[\s‌-]*سکه|ربع[\s‌-]*سکه|نقره|یورو|درهم|بیت[\s‌-]*کوین|نفت|"
    r"نرخ\s*ارز|دلار\s*به\s*ریال|ریال.*دلار|دلار.*ریال|"
    r"ecotrust|اکوتراست)",
    re.I,
)


def is_dollar_rial_question(text: str) -> bool:
    """True for dollar / gold / coin / FX questions answered via ecotrust."""
    return bool(_MARKET_RE.search(text or ""))


# keep old name used by fastapi
is_market_question = is_dollar_rial_question


def call_usd_irr(question: str = "قیمت دلار", **_: Any) -> dict[str, Any]:
    from ecotrust_prices.service import get_market_price

    data = get_market_price(question)
    return {
        "tool": "ecotrust_get_market_price",
        "text": data["answer_fa"],
        "raw": data,
        "url": data.get("source_url") or "https://ecotrust.ir/",
    }


if __name__ == "__main__":
    import json

    q = sys.argv[1] if len(sys.argv) > 1 else "قیمت دلار امروز"
    print(json.dumps(call_usd_irr(q), ensure_ascii=False, indent=2))
