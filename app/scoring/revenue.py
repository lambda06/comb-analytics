import math
from app.models import SignalResult
from app.scoring.confidence import time_series_confidence

def score_revenue(revenue_by_month: dict[str, float] | None) -> SignalResult:
    if not revenue_by_month or len(revenue_by_month) < 2:
        return SignalResult(score=0.0, confidence=0.0, factors={})

    months = list(revenue_by_month.keys())
    vals = list(revenue_by_month.values())
    n_months = len(vals)
    
    # Calculate MoM percentage deltas
    deltas = [(vals[i] - vals[i-1]) / vals[i-1] * 100.0 for i in range(1, n_months)]
    
    # Linear decay weighting
    weights = []
    n_deltas = len(deltas)
    for i in range(n_deltas):
        months_ago = (n_deltas - 1) - i
        w = max(0.1, 1.0 - 0.1 * months_ago)
        weights.append(w)
        
    total_w = sum(weights)
    weighted_avg = sum(d * w for d, w in zip(deltas, weights)) / total_w
    
    # Unweighted for comparison in explanation
    unweighted_avg = sum(deltas) / len(deltas)
    
    # Sigmoid mapping: maps weighted MoM growth → score in (0, 100)
    # k=0.12 calibration:
    score = 100.0 / (1.0 + math.exp(-0.12 * weighted_avg))
    score = round(score, 1)
    confidence = time_series_confidence(n_months)
    
    factors = {
        "n_months": n_months,
        "weighted_mom_growth": round(weighted_avg, 2),
        "unweighted_mom_growth": round(unweighted_avg, 2),
        "latest_mom_change": round(deltas[-1], 2),
        "peak_revenue": max(vals),
        "latest_revenue": vals[-1]
    }
    
    return SignalResult(score=round(score, 1), confidence=confidence, factors=factors)
