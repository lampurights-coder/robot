"""USD price in Iranian Rial (IRR) — TGJU first, web search fallback. No email."""

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

from dollar_rial.models import PriceSource, UsdIrrResult

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"), override=False)

TGJU_PAGE = "https://www.tgju.org/profile/price_dollar_rl"
TGJU_API = (
    "https://api.tgju.org/v1/market/indicator/summary-table-data/price_dollar_rl"
)

_IRAN_HOSTS = (
    "tgju.org",
    "bonbast.com",
    "navasan.net",
    "nobitex.ir",
    "mesghal.com",
    "tala.ir",
    "eghtesadnews.com",
    "isna.ir",
    "mehrnews.com",
)


def _parse_irr_number(text: str) -> float | None:
    """Parse '2,641,050' or '۲۶۴۱۰۵۰' style strings to float Rials."""
    if not text:
        return None
    # Persian digits → Latin
    trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    t = text.translate(trans)
    t = re.sub(r"[^\d.]", "", t.replace(",", ""))
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _from_tgju_api() -> UsdIrrResult | None:
    try:
        r = requests.get(
            TGJU_API,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DollarRialTool/1.0)"},
        )
        r.raise_for_status()
        rows = r.json().get("data") or []
        if not rows:
            return None
        # First row = latest: [open?, …] — TGJU uses [p, …] ; observed:
        # ["2,641,050","2,616,000","2,651,000","2,626,000", ...]
        cells = rows[0]
        candidates = [_parse_irr_number(c) for c in cells[:4] if isinstance(c, str)]
        candidates = [c for c in candidates if c and 500_000 <= c <= 5_000_000]
        if not candidates:
            return None
        # Prefer first cell as last/current-ish
        rate = candidates[0]
        return UsdIrrResult(
            answer_fa=(
                f"طبق داده‌های شبکه اطلاع‌رسانی طلا و ارز (tgju)، "
                f"قیمت دلار آزاد حدود {rate:,.0f} ریال است."
            ).replace(",", "٬"),
            usd_irr=rate,
            market="آزاد",
            notes_fa="منبع: TGJU API",
            source_url=TGJU_PAGE,
            sources=[PriceSource(title="قیمت دلار آزاد | TGJU", url=TGJU_PAGE)],
            fetched_at=datetime.now(timezone.utc),
        )
    except Exception:
        return None


def _from_tgju_page() -> UsdIrrResult | None:
    try:
        html = requests.get(
            TGJU_PAGE,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DollarRialTool/1.0)"},
        ).text
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        # look for large IRR-like numbers near دلار
        nums = re.findall(r"(\d{1,3}(?:,\d{3}){2,})", text)
        parsed = [_parse_irr_number(n) for n in nums]
        parsed = [p for p in parsed if p and 500_000 <= p <= 5_000_000]
        if not parsed:
            return None
        rate = parsed[0]
        return UsdIrrResult(
            answer_fa=(
                f"طبق صفحه TGJU، قیمت دلار آزاد حدود {rate:,.0f} ریال است."
            ).replace(",", "٬"),
            usd_irr=rate,
            market="آزاد",
            source_url=TGJU_PAGE,
            sources=[PriceSource(title="قیمت دلار آزاد | TGJU", url=TGJU_PAGE)],
            fetched_at=datetime.now(timezone.utc),
        )
    except Exception:
        return None


def _langsearch(query: str, count: int = 8) -> list[PriceSource]:
    key = os.environ.get("LANGSEARCH_API_KEY")
    if not key:
        raise RuntimeError("LANGSEARCH_API_KEY is missing in agent/.env")
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
            "freshness": "oneDay",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        PriceSource(title=item.get("name") or "", url=item["url"])
        for item in payload["data"]["webPages"]["value"]
    ]


def _pick_iran(sources: list[PriceSource]) -> PriceSource | None:
    scored: list[tuple[int, PriceSource]] = []
    for s in sources:
        blob = f"{s.title} {s.url}".lower()
        # skip obvious non-Iran FX pages
        if any(
            x in blob
            for x in (
                "rupee",
                "ringgit",
                "egypt",
                "guinea",
                "baht",
                "naira",
                "india",
                "malaysia",
            )
        ):
            continue
        score = 0
        if any(h in blob for h in _IRAN_HOSTS):
            score += 15
        if any(w in blob for w in ("ایران", "ریال", "دلار آزاد", "tgju", "bonbast")):
            score += 8
        if "irr" in blob or "rial" in blob:
            score += 4
        scored.append((score, s))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored[0][0] < 4:
        return None
    return scored[0][1]


def _fetch_page_text(url: str, max_chars: int = 10000) -> str:
    html = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DollarRialTool/1.0)"},
    ).text
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)[:max_chars]


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1"),
        temperature=0,
    )


def _structure_with_llm(
    page_text: str,
    source_url: str,
    sources: list[PriceSource],
) -> UsdIrrResult:
    llm = _llm().with_structured_output(UsdIrrResult)
    prompt = f"""Extract TODAY's US Dollar price in Iranian Rials (IRR / ریال ایران) from the page.

Rules:
- This is IRAN (ریال ایران), NOT Indian Rupee, NOT other currencies.
- answer_fa MUST be Persian
- usd_irr = free-market (بازار آزاد) rate in RIALS. If page shows Tomans, multiply ×10.
- Typical free market is hundreds of thousands to a few million Rials — reject tiny numbers.
- market = آزاد / نیما / رسمی
- source_url="{source_url}"

Sources:
{json.dumps([s.model_dump() for s in sources], ensure_ascii=False)}

Page text:
{page_text}
"""
    return llm.invoke(prompt)


def get_usd_irr_price(*, force_refresh: bool = False) -> UsdIrrResult:
    """Get current USD→IRR. Prefer TGJU; else Iran-focused web search."""
    del force_refresh

    hit = _from_tgju_api() or _from_tgju_page()
    if hit is not None:
        return hit

    sources: list[PriceSource] = []
    for q in (
        "site:tgju.org قیمت دلار آزاد",
        "قیمت دلار آزاد امروز ریال ایران tgju",
        "USD IRR Iran free market today tgju bonbast",
    ):
        try:
            sources = _langsearch(q)
            if sources:
                break
        except Exception:
            continue
    if not sources:
        raise RuntimeError("No web results for Iranian USD/IRR price")

    best = _pick_iran(sources) or PriceSource(title="TGJU", url=TGJU_PAGE)
    try:
        page_text = _fetch_page_text(best.url)
    except Exception:
        page_text = " | ".join(f"{s.title}" for s in sources)

    try:
        result = _structure_with_llm(page_text, best.url, sources)
    except (ValidationError, Exception):
        # last resort numbers from page
        nums = [
            _parse_irr_number(n)
            for n in re.findall(r"(\d{1,3}(?:,\d{3}){2,})", page_text)
        ]
        nums = [n for n in nums if n and 500_000 <= n <= 5_000_000]
        rate = nums[0] if nums else None
        result = UsdIrrResult(
            answer_fa=(
                f"بر اساس منابع وب، قیمت دلار حدود {rate:,.0f} ریال است.".replace(",", "٬")
                if rate
                else "نتوانستیم قیمت دقیق دلار به ریال را استخراج کنیم."
            ),
            usd_irr=rate,
            market="آزاد",
            source_url=best.url,
            sources=sources,
            fetched_at=datetime.now(timezone.utc),
        )

    result.source_url = best.url
    result.sources = sources or result.sources
    result.fetched_at = datetime.now(timezone.utc)
    return result


if __name__ == "__main__":
    print(get_usd_irr_price().model_dump_json(indent=2, ensure_ascii=False))
