"""Weather MCP: web search once, cache page URL, return Pydantic weather results."""

from weather_mcp.models import WeatherResult
from weather_mcp.service import get_weather

__all__ = ["WeatherResult", "get_weather"]
