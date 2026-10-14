"""Scrape live market prices from https://ecotrust.ir/ (no web search, no email).

Parses the landing-page embedded currency JSON (same data the site shows for
دلار/تتر، طلا، سکه، یورو، …).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

ECOTRUST_URL = "https://ecotrust.ir/"

# map user intents → preferred symbol substrings (order = preference)
_ALIASES: dict[str, tuple[str, ...]] = {
    "dollar": ("تتر", "دلار مجازی", "USDT", "دلار"),
    "usdt": ("تتر", "دلار مجازی", "USDT"),
    "gold18": ("طلای ۱۸", "طلای 18", "GOLD18", "عیار"),
    "gold_ounce": ("اونس طلا", "GOLD-OUNCE", "اونس"),
    "seke_emami": ("سکه امامی", "امامی"),
    "seke_bahar": ("سکه بهار", "بهار آزادی"),
    "nim": ("نیم‌سکه", "نیم سکه", "نیمسکه"),
    "rob": ("ربع‌سکه", "ربع سکه", "ربעסکه"),
    "silver": ("نقره", "SILVER"),
    "euro": ("یورو", "EUR", "euro"),
    "aed": ("درهم", "AED"),
    "btc": ("بیت‌کوین", "بیت کوین", "BTC", "bitcoin"),
    "oil": ("نفت", "OIL", "برنت"),
}


@dataclass
class MarketItem:
    symbol: str
    value: float
    quote: str  # تومان / USD / USDT
    change: str | None
    date: str | None
    source_url: str = ECOTRUST_URL

    @property
    def value_rial(self) -> float | None:
        """If quote is تومان, return ریال (= ×10)."""
        if "تومان" in (self.quote or ""):
            return self.value * 10
        return None


def _fetch_html() -> str:
    r = requests.get(
        ECOTRUST_URL,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; EcotrustPriceTool/1.0)",
            "Accept-Language": "fa-IR,fa;q=0.9",
        },
    )
    r.raise_for_status()
    return r.text


def _parse_embedded_json(html: str) -> list[MarketItem]:
    """Unescape JS-embedded JSON snippets and pull symbol + latest_price."""
    # normalize escaped quotes used inside Next.js payload strings
    blob = html.replace('\\"', '"')
    items: list[MarketItem] = []
    seen: set[str] = set()
    # Flexible: fields between symbol and latest_price may vary (price_source_key, …)
    pat = re.compile(
        r'"symbol"\s*:\s*"([^"]+)"'
        r'(?:(?!"symbol").)*?'
        r'"latest_price"\s*:\s*\{'
        r'"change"\s*:\s*"([^"]*)"\s*,\s*"bubble"\s*:\s*[^,]*\s*,\s*"date"\s*:\s*"([^"]*)"'
        r'\s*,\s*"value"\s*:\s*"([^"]+)"\s*\}'
        r'(?:(?!"symbol").)*?'
        r'"fa"\s*:\s*\{\s*"quote"\s*:\s*"([^"]+)"\s*\}',
        re.S,
    )
    for m in pat.finditer(blob):
        symbol, change, date, value_s, quote = m.groups()
        try:
            value = float(value_s)
        except ValueError:
            continue
        key = f"{symbol}|{value}"
        if key in seen:
            continue
        seen.add(key)
        items.append(
            MarketItem(
                symbol=symbol,
                value=value,
                quote=quote,
                change=change or None,
                date=date or None,
            )
        )
    return items


def _parse_dom_fallback(html: str) -> list[MarketItem]:
    """Fallback: label + number + تومان from visible text."""
    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    lines = text.split("\n")
    items: list[MarketItem] = []
    for i, line in enumerate(lines):
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", line.replace("٬", ",")):
            continue
        label = lines[i - 1] if i else ""
        unit = lines[i + 1] if i + 1 < len(lines) else "تومان"
        if not any(
            k in label
            for k in ("طلا", "سکه", "تتر", "یورو", "درهم", "نقره", "بیت", "اونس", "نفت", "دلار")
        ):
            continue
        val = float(line.replace(",", "").replace("٬", ""))
        items.append(
            MarketItem(
                symbol=label,
                value=val,
                quote="تومان" if "تومان" in unit else unit,
                change=None,
                date=None,
            )
        )
    return items


def list_market_prices() -> list[MarketItem]:
    html = _fetch_html()
    items = _parse_embedded_json(html)
    if not items:
        items = _parse_dom_fallback(html)
    if not items:
        raise RuntimeError("Could not parse prices from ecotrust.ir")
    return items


def _detect_key(question: str) -> str:
    q = (question or "").lower()
    # Persian / English intents
    rules: list[tuple[str, tuple[str, ...]]] = [
        ("usdt", ("تتر", "usdt", "دلار مجازی")),
        ("dollar", ("دلار", "dollar", "usd", "قیمت دلار")),
        ("gold18", ("طلای ۱۸", "طلای 18", "طلا ۱۸", "عیار ۱۸", "gold18", "قیمت طلا")),
        ("gold_ounce", ("اونس", "ounce")),
        ("seke_emami", ("امامی",)),
        ("seke_bahar", ("بهار",)),
        ("nim", ("نیم",)),
        ("rob", ("ربع",)),
        ("silver", ("نقره", "silver")),
        ("euro", ("یورو", "euro", "eur")),
        ("aed", ("درهم", "aed", "dirham")),
        ("btc", ("بیت", "bitcoin", "btc")),
        ("oil", ("نفت", "brent", "oil")),
    ]
    for key, words in rules:
        if any(w in q for w in words):
            return key
    return "dollar"


def _match_item(items: list[MarketItem], key: str) -> MarketItem:
    aliases = _ALIASES.get(key, _ALIASES["dollar"])
    for alias in aliases:
        for it in items:
            if alias.lower() in it.symbol.lower():
                return it
    # last resort: first item
    return items[0]


def get_market_price(question: str) -> dict:
    """Return one ecotrust price matching the user question."""
    items = list_market_prices()
    key = _detect_key(question)
    it = _match_item(items, key)
    rial = it.value_rial
    change = f" ({it.change}%)" if it.change else ""
    if rial is not None:
        answer = (
            f"طبق اکوتراست (ecotrust.ir)، قیمت «{it.symbol}» "
            f"{it.value:,.0f} {it.quote} است"
            f"{change} — معادل حدود {rial:,.0f} ریال."
        ).replace(",", "٬")
    else:
        answer = (
            f"طبق اکوتراست (ecotrust.ir)، قیمت «{it.symbol}» "
            f"{it.value:,.4g} {it.quote} است{change}."
        )
    return {
        "answer_fa": answer,
        "symbol": it.symbol,
        "value": it.value,
        "value_rial": rial,
        "quote": it.quote,
        "change": it.change,
        "date": it.date,
        "source_url": ECOTRUST_URL,
        "key": key,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "all_symbols": [x.symbol for x in items],
    }


if __name__ == "__main__":
    for q in ("قیمت دلار", "قیمت طلا", "سکه امامی", "یورو"):
        print(json.dumps(get_market_price(q), ensure_ascii=False, indent=2))
        print("---")
