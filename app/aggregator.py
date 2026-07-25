"""
Aggregator: combines the 4 signal scores into a single composite.

Formula:
    composite = Σ(score_i × confidence_i) / Σ(confidence_i)

Signals with confidence == 0.0 are excluded from BOTH the numerator
and denominator — they abstain rather than penalise or inflate.
"""

from __future__ import annotations

from app.models import Business, BusinessScoreResponse, SignalScore
from app.scoring.revenue import score_revenue
from app.scoring.reviews import score_reviews
from app.scoring.retention import score_retention_and_support
from app.scoring.ad_efficiency import score_ad_efficiency

# Any included signal with confidence below this threshold triggers
# a low_confidence_warning on the response.
LOW_CONFIDENCE_THRESHOLD = 0.35


def score_business(biz: Business) -> BusinessScoreResponse:
    """
    Run all four signal scorers on a Business and return the full response.
    """
    raw = {
        "revenue": score_revenue(biz.revenue_by_month),
        "reviews": score_reviews(biz.customer_reviews),
        "retention_support": score_retention_and_support(
            biz.repeat_purchase_rate, biz.customer_support_tickets
        ),
        "ad_efficiency": score_ad_efficiency(
            biz.revenue_by_month, biz.ad_spend_by_month
        ),
    }

    # Partition into included (confidence > 0) and excluded (confidence == 0)
    included: dict[str, SignalScore] = {}
    excluded: list[str] = []

    for name, result in raw.items():
        if result.confidence == 0.0:
            excluded.append(name)
        else:
            included[name] = SignalScore(
                score=result.score,
                confidence=result.confidence,
                factors=result.factors,
            )

    # Confidence-weighted composite
    numerator = sum(s.score * s.confidence for s in included.values())
    denominator = sum(s.confidence for s in included.values())

    composite = round(numerator / denominator, 1) if denominator > 0 else 0.0

    # Warn if any included signal is thin
    low_conf = any(
        s.confidence < LOW_CONFIDENCE_THRESHOLD for s in included.values()
    )

    return BusinessScoreResponse(
        business_id=biz.business_id,
        business_name=biz.business_name,
        category=biz.category,
        composite_score=composite,
        signal_scores=included,
        excluded_signals=excluded,
        low_confidence_warning=low_conf,
    )
