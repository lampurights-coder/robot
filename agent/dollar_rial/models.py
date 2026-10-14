"""Pydantic models for USD → Iranian Rial price tool."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PriceSource(BaseModel):
    title: str
    url: str


class UsdIrrResult(BaseModel):
    """Structured USD/IRR quote (Persian answer, rates in Iranian Rials)."""

    answer_fa: str = Field(description="Full answer in Persian about dollar price in Rials")
    usd_irr: float | None = Field(
        default=None,
        description="1 USD in Iranian Rials (free market / common rate if available)",
    )
    usd_irr_official: float | None = Field(
        default=None,
        description="Official / NIMA / other official rate in Rials if mentioned",
    )
    market: str | None = Field(
        default=None,
        description="Which market: آزاد / نیما / رسمی / …",
    )
    notes_fa: str | None = None
    source_url: str = ""
    sources: list[PriceSource] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=_utc_now)
