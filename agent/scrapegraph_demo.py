"""Search → scrape → email the result via Gmail."""

import html
import json
import os
from pathlib import Path

import nest_asyncio
import requests
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph

from gmail_tool import send_gmail

nest_asyncio.apply()
load_dotenv(override=True)

CHROME = str(
    Path.home() / ".cache/ms-playwright/chromium-1243/chrome-linux64/chrome"
)


class QuotaError(RuntimeError):
    """GapGPT / OpenAI-compatible provider has no remaining credit."""


def langsearch(query: str, count: int = 5) -> list[dict]:
    response = requests.post(
        "https://api.langsearch.com/v1/web-search",
        headers={
            "Authorization": f"Bearer {os.environ['LANGSEARCH_API_KEY']}",
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
        {
            "title": item.get("name"),
            "url": item["url"],
            "context": item.get("text") or item.get("snippet") or "",
        }
        for item in payload["data"]["webPages"]["value"]
    ]


def _raise_if_quota(err: Exception) -> None:
    msg = str(err).lower()
    if "insufficient_user_quota" in msg or "pre-consume quota failed" in msg:
        raise QuotaError(
            "GapGPT quota is too low for ScrapeGraphAI.\n"
            "  Top up at https://gapgpt.app or set OPENAI_MODEL=gpt-4o-mini in .env."
        ) from err


def scrape_url(url: str, prompt: str):
    graph_config = {
        "llm": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": f"openai/{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}",
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1"),
        },
        "verbose": True,
        "headless": True,
        "loader_kwargs": {
            "executable_path": CHROME,
            "timeout": 60,
        },
    }
    smart = SmartScraperGraph(prompt=prompt, source=url, config=graph_config)
    try:
        return smart.run()
    except Exception as e:
        _raise_if_quota(e)
        raise


def scrape_text_as_html(text: str, prompt: str, tmp_path: Path):
    safe = html.escape(text[:12000])
    tmp_path.write_text(
        f"<html><body><article><p>{safe}</p></article></body></html>",
        encoding="utf-8",
    )
    return scrape_url(str(tmp_path), prompt)


def research(query: str) -> tuple[list[dict], object, str]:
    """Search first, then scrape. Returns (sources, scrape_result, used_url)."""
    prompt = (
        f"Using only the page content, answer this question: {query}\n\n"
        "IMPORTANT rules:\n"
        "1) Write ALL text answers in Persian (Farsi).\n"
        "2) Report ALL temperatures in Celsius (°C / سانتی‌گراد). "
        "If the page uses Fahrenheit, convert to Celsius.\n"
        "3) Return JSON with fields: "
        "answer_fa (full Persian answer), "
        "temp_high_c, temp_low_c, conditions_fa, air_quality_fa, notes_fa.\n"
        "Do not use Fahrenheit in the output."
    )

    # Search in English for better web hits; answer still in Farsi °C
    search_query = query
    if any("\u0600" <= ch <= "\u06FF" for ch in query):
        search_query = (
            "Tehran weather today forecast temperature Celsius"
            if "تهران" in query or "آب" in query
            else query
        )
        print(f"1) Search query (English): {search_query}")
    else:
        print(f"1) Search query: {search_query}")

    sources = langsearch(search_query)
    print("   Found URLs:")
    for s in sources:
        print("  -", s["url"])

    result = None
    used_url = ""
    for s in sources:
        print("\n2) Scrape — trying:", s["url"])
        try:
            result = scrape_url(s["url"], prompt)
            used_url = s["url"]
            break
        except QuotaError:
            raise
        except Exception as e:
            print("  Failed:", e)

    if result is None:
        print("\n2) Scrape — all pages failed; using LangSearch snippets...")
        combined = "\n\n".join(
            f"Title: {s['title']}\nURL: {s['url']}\n{s['context']}"
            for s in sources
            if s.get("context")
        )
        if not combined.strip():
            raise RuntimeError("No LangSearch context available to fall back on.")
        result = scrape_text_as_html(
            combined, prompt, Path("/tmp/langsearch_fallback.html")
        )
        used_url = "langsearch snippets (pages unreachable)"

    return sources, result, used_url


def _pick(data: dict, *keys, default="—"):
    for key in keys:
        if key in data and data[key] not in (None, "", "NA", "N/A"):
            return data[key]
    return default


def build_email(query: str, sources: list[dict], result: object, used_url: str):
    """Build Farsi email subject/body; temperatures expected in °C."""
    subject = f"گزارش آب‌وهوا: {query[:60]}"

    content = result
    if isinstance(result, dict) and "content" in result:
        content = result["content"]

    if isinstance(content, dict):
        answer = _pick(content, "answer_fa", "weather_today", "summary")
        high = _pick(content, "temp_high_c", "high_c")
        low = _pick(content, "temp_low_c", "low_c")
        conditions = _pick(content, "conditions_fa", "conditions")
        air = _pick(content, "air_quality_fa", "air_quality")
        notes = _pick(content, "notes_fa", "notes", default="")
        summary = f"""پاسخ:
{answer}

حداکثر دما: {high} °C
حداقل دما: {low} °C
وضعیت: {conditions}
کیفیت هوا: {air}
"""
        if notes and notes != "—":
            summary += f"یادداشت: {notes}\n"
    else:
        summary = str(result)

    links = "\n".join(f"- {s['title']}: {s['url']}" for s in sources)

    body = f"""سوال: {query}

منبع خوانده‌شده: {used_url}

=== خلاصه (فارسی، سانتی‌گراد) ===
{summary}
=== منابع ===
{links}
"""
    return subject, body


if __name__ == "__main__":
    if not Path(CHROME).exists():
        raise SystemExit(
            f"Chrome not found at {CHROME}\n"
            "Unzip chrome-linux64.zip into ~/.cache/ms-playwright/chromium-1243/"
        )

    query = os.getenv("RESEARCH_QUERY", "is ayatollah khamenei alive?")
    to = os.getenv("EMAIL_TO", "larshiakazemil.lll@gmail.com")

    try:
        sources, result, used_url = research(query)
    except QuotaError as e:
        print("\nSTOP:", e)
        raise SystemExit(1) from e

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, (dict, list)) else result)

    subject, body = build_email(query, sources, result, used_url)
    print("\n3) Email")
    print("  To:", to)
    print("  Subject:", subject)

    status = send_gmail.invoke({"to": to, "subject": subject, "body": body})
    print(" ", status)
