"""Weather-only MCP server.

Tools:
  - get_weather: search (or reuse cached URL) and return Pydantic WeatherResult
  - list_weather_cache / clear_weather_cache: manage URL cache
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python weather_mcp/server.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(ROOT / ".env", override=True)

from weather_mcp import cache
from weather_mcp.models import CacheActionResult, WeatherResult
from weather_mcp.service import get_weather

mcp = FastMCP(
    "weather",
    instructions=(
        "Weather MCP only. Use get_weather(city) for today's weather. "
        "Answers are Persian with Celsius. Cached city URLs skip web search."
    ),
)


@mcp.tool()
def get_weather_tool(
    city: str,
    force_refresh: bool = False,
    force_search: bool = False,
) -> WeatherResult:
    """Get today's weather for a city.

    First call searches the web and caches the best weather page URL.
    Later calls reuse that URL (no search) unless force_search=True.
    Same-day structured results are returned from cache unless force_refresh=True.
    Returns Pydantic WeatherResult (Farsi text, temps in °C).
    """
    return get_weather(
        city,
        force_refresh=force_refresh,
        force_search=force_search,
    )


@mcp.tool()
def list_weather_cache() -> CacheActionResult:
    """List cities that have a cached weather page URL."""
    cities = cache.list_cities()
    return CacheActionResult(
        status="listed",
        message=f"{len(cities)} cached cities",
        cities=cities,
    )


@mcp.tool()
def clear_weather_cache(city: str | None = None) -> CacheActionResult:
    """Clear cached weather URL/result for one city, or all if city is omitted."""
    cleared = cache.clear(city)
    if city and not cleared:
        return CacheActionResult(
            status="missing",
            city=city,
            message=f"No cache for {city}",
        )
    return CacheActionResult(
        status="cleared",
        city=city,
        message=f"Cleared: {', '.join(cleared) or 'nothing'}",
        cities=cleared,
    )


if __name__ == "__main__":
    mcp.run()
