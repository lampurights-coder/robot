"""Simple search + page fetch + LLM answer. No Playwright / ScrapeGraphAI."""

import json
import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1"),
)


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


def fetch_page(url: str, max_chars: int = 8000) -> str:
    html = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    ).text
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)[:max_chars]


def answer_question(query: str) -> str:
    sources = langsearch(query)
    print("Found URLs:")
    for s in sources:
        print("-", s["url"])

    url = sources[0]["url"]
    try:
        page_text = fetch_page(url)
    except Exception as e:
        print("Fetch failed, using LangSearch context:", e)
        page_text = sources[0].get("context", "")

    prompt = f"""Based on these sources, answer: {query}

Source URL: {url}
Page text:
{page_text}

Also use these search snippets:
{json.dumps(sources, ensure_ascii=False, indent=2)}

Give a short answer with conflicting claims and cite URLs.
"""
    return llm.invoke(prompt).content


if __name__ == "__main__":
    print("\n=== ANSWER ===")
    print(answer_question("is ayatollah khamenei alive?"))
