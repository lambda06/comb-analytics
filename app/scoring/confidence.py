"""
Shared confidence curves used by all signal scorers.

Each function returns a float in [0, 1] representing how much weight
the scoring system should assign to a signal given its data richness.

Design rationale:
- Confidence is a function of DATA AVAILABILITY (is the field present?),
  DATA VOLUME (how many data points?), and implicitly DATA RECENCY
  (handled separately in the revenue scorer via weighted averages).
- We use simple bracket-based curves rather than continuous functions
  because they are transparent.
  The brackets were chosen to reflect natural thresholds: 2 months is
  barely a trend, 6 months shows a pattern, 10+ months is reliable.
"""

from __future__ import annotations


def time_series_confidence(n_points: int) -> float:
    """
    Confidence curve for any monthly time-series signal (revenue, ad spend).

    n_points is the number of MONTHS of data (not deltas).
    We need at least 2 months to compute any MoM delta, so n=1 → caller
    should return None (signal excluded).
    """
    if n_points < 2:
        return 0.0   # caller treats as excluded
    if n_points < 3:
        return 0.25  # 2 months: 1 delta, very thin
    if n_points < 6:
        return 0.50  # 3–5 months: emerging trend
    if n_points < 10:
        return 0.75  # 6–9 months: solid pattern
    return 0.95      # 10+ months: highly reliable


def review_confidence(n_reviews: int) -> float:
    """
    Confidence curve for the customer review signal.

    Note: n_reviews == 0 (empty list []) is different from None (excluded).
    An empty list means we have the signal category but zero data points —
    it gets a tiny confidence (0.05) rather than being fully excluded.
    """
    if n_reviews == 0:
        return 0.05  # signal present, zero data — near-excluded but not fully
    if n_reviews <= 3:
        return 0.25
    if n_reviews <= 7:
        return 0.55
    return 0.90


def rpr_sub_confidence(rpr: float | None) -> float:
    """Sub-confidence for the RPR component of the retention signal."""
    return 0.0 if rpr is None else 1.0


def support_sub_confidence(total_tickets: int | None) -> float:
    """
    Sub-confidence for the support-ticket component of the retention signal.

    total_tickets=None  → tickets object was null → 0.0
    total_tickets=0     → object present, no tickets → 0.10 (no evidence either way)
    total_tickets 1–19  → thin sample → 0.50
    total_tickets ≥ 20  → solid sample → 0.85
    """
    if total_tickets is None:
        return 0.0
    if total_tickets == 0:
        return 0.10
    if total_tickets < 20:
        return 0.50
    return 0.85
