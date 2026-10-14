"""Pydantic models for the weather MCP."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WeatherSource(BaseModel):
    title: str
    url: str


class WeatherResult(BaseModel):
    """Structured weather answer (Persian text, Celsius temps)."""

    city: str
    answer_fa: str = Field(description="Full answer in Persian (Farsi)")
    temp_high_c: float | None = Field(default=None, description="High temp °C")
    temp_low_c: float | None = Field(default=None, description="Low temp °C")
    conditions_fa: str | None = None
    air_quality_fa: str | None = None
    notes_fa: str | None = None
    source_url: str
    sources: list[WeatherSource] = Field(default_factory=list)
    cache_hit: bool = Field(
        default=False,
        description="True if a cached weather page URL was reused (no web search)",
    )
    fetched_at: datetime = Field(default_factory=_utc_now)


class CacheEntry(BaseModel):
    """Cached preferred weather page for a city."""

    city_key: str
    url: str
    title: str = ""
    last_result: WeatherResult | None = None
    updated_at: datetime = Field(default_factory=_utc_now)


class CacheStore(BaseModel):
    entries: dict[str, CacheEntry] = Field(default_factory=dict)


class GetWeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. Tehran / تهران")
    force_refresh: bool = Field(
        default=False,
        description="If true, ignore cached result and re-fetch the page (still skips search if URL is cached)",
    )
    force_search: bool = Field(
        default=False,
        description="If true, run a new web search even if a URL is cached",
    )


class CacheActionResult(BaseModel):
    status: Literal["cleared", "missing", "listed"]
    city: str | None = None
    message: str
    cities: list[str] = Field(default_factory=list)
