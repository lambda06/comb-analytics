from app.models import SignalResult
from app.scoring.confidence import time_series_confidence

def score_ad_efficiency(revenue_by_month: dict[str, float] | None, ad_spend_by_month: dict[str, float] | None) -> SignalResult:
    if not revenue_by_month or not ad_spend_by_month:
        return SignalResult(score=0.0, confidence=0.0, factors={})
        
    # Find overlapping months
    overlap = sorted(set(revenue_by_month.keys()) & set(ad_spend_by_month.keys()))
    if len(overlap) < 1:
        return SignalResult(score=0.0, confidence=0.0, factors={})
        
    roas_vals = []
    for m in overlap:
        rev = revenue_by_month[m]
        spend = ad_spend_by_month[m]
        if spend > 0:
            roas_vals.append(rev / spend)
            
    if not roas_vals:
        return SignalResult(score=0.0, confidence=0.0, factors={})
        
    n_months = len(overlap)
    confidence = time_series_confidence(n_months)
    
    avg_roas = sum(roas_vals) / len(roas_vals)
    latest_roas = roas_vals[-1]
    
    # Cap ROAS at 20x for scoring
    score = min(avg_roas / 20.0, 1.0) * 100.0
    
    # Trend penalty if latest ROAS is significantly lower than average
    trend = "stable"
    if len(roas_vals) >= 2:
        if latest_roas < avg_roas * 0.8:
            trend = "declining"
            score -= 8.0
        elif latest_roas > avg_roas * 1.2:
            trend = "improving"
            score += 5.0
            
    score = max(0.0, min(100.0, score))
    
    factors = {
        "n_months": n_months,
        "avg_roas": round(avg_roas, 1),
        "latest_roas": round(latest_roas, 1),
        "roas_trend": trend
    }
    
    return SignalResult(score=round(score, 1), confidence=confidence, factors=factors)
