"""Weather search + page fetch + structured Persian/Celsius answer."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from weather_mcp import cache
from weather_mcp.models import WeatherResult, WeatherSource

load_dotenv(override=True)

F_TO_C = lambda f: round((f - 32) * 5 / 9, 1)


def _langsearch(query: str, count: int = 8) -> list[WeatherSource]:
    key = os.environ.get("LANGSEARCH_API_KEY")
    if not key:
        raise RuntimeError("LANGSEARCH_API_KEY is missing in .env")

    response = requests.post(
        "https://api.langsearch.com/v1/web-search",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "count": count,
            "contents": {"text": True},
            "freshness": "noLimit",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        WeatherSource(
            title=item.get("name") or "",
            url=item["url"],
        )
        for item in payload["data"]["webPages"]["value"]
    ]


_WEATHER_HOST_BONUS = (
    "theweather.com",
    "weather.com",
    "accuweather.com",
    "weather-forecast.com",
    "timeanddate.com",
    "wunderground.com",
    "yr.no",
    "weather-and-climate.com",
    "yourweather",
)


def _pick_best_source(city: str, sources: list[WeatherSource]) -> WeatherSource:
    """Prefer real weather sites that mention the city."""
    needle = city.strip().lower()
    aliases = {needle}
    if needle in {"tehran", "تهران"}:
        aliases.update({"tehran", "تهران"})

    scored: list[tuple[int, WeatherSource]] = []
    for s in sources:
        blob = f"{s.title} {s.url}".lower()
        score = 0
        if any(a in blob for a in aliases):
            score += 8
        if any(h in blob for h in _WEATHER_HOST_BONUS):
            score += 10
        if "weather" in blob or "forecast" in blob:
            score += 3
        # Penalize unrelated news
        if any(w in blob for w in ("war", "iran war", "hormuz", "politics", "mediate")):
            score -= 6
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    # If search is junk, fall back to a known weather URL template
    if scored[0][0] < 8 and needle in {"tehran", "تهران"}:
        return WeatherSource(
            title="Tehran Weather",
            url="https://www.theweather.com/tehran.htm",
        )
    return best


def _fetch_page_text(url: str, max_chars: int = 10000) -> str:
    html = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; WeatherMCP/1.0)"},
    ).text
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return text[:max_chars]


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1"),
        temperature=0,
    )


def _structure_with_llm(
    city: str,
    page_text: str,
    source_url: str,
    sources: list[WeatherSource],
    cache_hit: bool,
) -> WeatherResult:
    llm = _llm().with_structured_output(WeatherResult)
    prompt = f"""Extract today's weather for {city} from the page text.

Rules:
- answer_fa, conditions_fa, air_quality_fa, notes_fa MUST be Persian (Farsi)
- temp_high_c and temp_low_c MUST be Celsius. Convert from °F if needed.
- Set city="{city}", source_url="{source_url}", cache_hit={cache_hit}
- sources = the provided source list
- fetched_at = now UTC

Sources JSON:
{json.dumps([s.model_dump() for s in sources], ensure_ascii=False)}

Page text:
{page_text}
"""
    return llm.invoke(prompt)


def _fallback_result(
    city: str,
    page_text: str,
    source_url: str,
    sources: list[WeatherSource],
    cache_hit: bool,
) -> WeatherResult:
    """Heuristic fallback if LLM / quota fails."""
    f_temps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*°?\s*F\b", page_text, re.I)]
    c_temps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*°?\s*C\b", page_text, re.I)]
    if c_temps:
        high, low = max(c_temps[:4]), min(c_temps[:4])
    elif f_temps:
        converted = [F_TO_C(t) for t in f_temps[:4]]
        high, low = max(converted), min(converted)
    else:
        high = low = None

    snippet = page_text[:500]
    return WeatherResult(
        city=city,
        answer_fa=f"آب‌وهوای {city}: (استخراج ساده بدون مدل) {snippet}",
        temp_high_c=high,
        temp_low_c=low,
        conditions_fa=None,
        air_quality_fa=None,
        notes_fa="پاسخ کامل فارسی در دسترس نبود؛ خلاصه صفحه برگردانده شد.",
        source_url=source_url,
        sources=sources,
        cache_hit=cache_hit,
        fetched_at=datetime.now(timezone.utc),
    )


def get_weather(
    city: str,
    *,
    force_refresh: bool = False,
    force_search: bool = False,
) -> WeatherResult:
    """
    1) Reuse cached weather URL for this city when possible (skip search).
    2) Only search the web if no cache / force_search.
    3) Fetch page and return structured WeatherResult.
    """
    entry = cache.get_entry(city)

    # Same-day cached structured result (fastest path)
    if (
        entry
        and entry.last_result
        and not force_refresh
        and not force_search
        and entry.last_result.fetched_at.date() == datetime.now(timezone.utc).date()
    ):
        cached = entry.last_result.model_copy(update={"cache_hit": True})
        return cached

    sources: list[WeatherSource] = []
    cache_hit = False
    source_url = ""

    if entry and entry.url and not force_search:
        source_url = entry.url
        cache_hit = True
        sources = [WeatherSource(title=entry.title or city, url=entry.url)]
    else:
        query = f'"{city}" weather today forecast temperature Celsius'
        sources = _langsearch(query)
        if not sources:
            raise RuntimeError(f"No weather pages found for {city}")
        best = _pick_best_source(city, sources)
        source_url = best.url
        cache.upsert_url(city, best.url, best.title)

    try:
        page_text = _fetch_page_text(source_url)
    except Exception:
        # Cached URL dead → search again once
        if cache_hit:
            sources = _langsearch(f"{city} Iran weather today forecast")
            if not sources:
                raise
            best = _pick_best_source(city, sources)
            source_url = best.url
            cache.upsert_url(city, best.url, best.title)
            cache_hit = False
            page_text = _fetch_page_text(source_url)
        else:
            raise

    try:
        result = _structure_with_llm(city, page_text, source_url, sources, cache_hit)
    except (ValidationError, Exception):
        result = _fallback_result(city, page_text, source_url, sources, cache_hit)

    result.cache_hit = cache_hit
    result.city = city
    result.source_url = source_url
    result.sources = sources or result.sources
    result.fetched_at = datetime.now(timezone.utc)
    cache.save_result(city, result)
    return result
