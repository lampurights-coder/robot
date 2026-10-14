"""Disk cache of preferred weather page URLs per city."""

from __future__ import annotations

import json
from pathlib import Path

from weather_mcp.models import CacheEntry, CacheStore, WeatherResult

CACHE_PATH = Path(__file__).resolve().parent / ".weather_url_cache.json"


def _normalize_city(city: str) -> str:
    return " ".join(city.strip().lower().split())


def load_store() -> CacheStore:
    if not CACHE_PATH.exists():
        return CacheStore()
    try:
        return CacheStore.model_validate_json(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return CacheStore()


def save_store(store: CacheStore) -> None:
    CACHE_PATH.write_text(
        store.model_dump_json(indent=2),
        encoding="utf-8",
    )


def get_entry(city: str) -> CacheEntry | None:
    store = load_store()
    return store.entries.get(_normalize_city(city))


def upsert_url(city: str, url: str, title: str = "") -> CacheEntry:
    store = load_store()
    key = _normalize_city(city)
    entry = store.entries.get(key) or CacheEntry(city_key=key, url=url, title=title)
    entry.url = url
    entry.title = title or entry.title
    store.entries[key] = entry
    save_store(store)
    return entry


def save_result(city: str, result: WeatherResult) -> None:
    store = load_store()
    key = _normalize_city(city)
    entry = store.entries.get(key)
    if entry is None:
        entry = CacheEntry(city_key=key, url=result.source_url, title="")
    entry.last_result = result
    entry.url = result.source_url
    store.entries[key] = entry
    save_store(store)


def clear(city: str | None = None) -> list[str]:
    store = load_store()
    if city is None:
        cleared = list(store.entries.keys())
        store.entries.clear()
        save_store(store)
        return cleared
    key = _normalize_city(city)
    if key in store.entries:
        del store.entries[key]
        save_store(store)
        return [key]
    return []


def list_cities() -> list[str]:
    return sorted(load_store().entries.keys())
