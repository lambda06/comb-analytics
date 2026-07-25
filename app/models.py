"""
Pydantic models for raw input data and API response schemas.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Input models (mirrors business_signals.json schema)
# ---------------------------------------------------------------------------

class Review(BaseModel):
    rating: int
    text: str


class SupportTickets(BaseModel):
    total: int
    resolved_within_48h: int
    escalated: int


class Business(BaseModel):
    business_id: str
    business_name: str
    category: str
    revenue_by_month: Optional[dict[str, float]] = None
    customer_reviews: Optional[list[Review]] = None
    repeat_purchase_rate: Optional[float] = None
    customer_support_tickets: Optional[SupportTickets] = None
    ad_spend_by_month: Optional[dict[str, float]] = None


class BusinessData(BaseModel):
    businesses: list[Business]


# ---------------------------------------------------------------------------
# Internal scoring types (not exposed directly in API)
# ---------------------------------------------------------------------------

class SignalResult(BaseModel):
    """Returned by each signal scorer. score=None means signal was excluded."""
    score: float          # 0–100
    confidence: float     # 0–1
    factors: dict         # raw computed numbers; passed verbatim to the LLM


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------

class SignalScore(BaseModel):
    score: float
    confidence: float
    factors: dict


class BusinessScoreResponse(BaseModel):
    business_id: str
    business_name: str
    category: str
    composite_score: float
    signal_scores: dict[str, SignalScore]
    excluded_signals: list[str]
    low_confidence_warning: bool   # true if any signal confidence < 0.35
    explanation: Optional[str] = None


class BusinessListItem(BaseModel):
    business_id: str
    business_name: str
    category: str


class HealthResponse(BaseModel):
    status: str
