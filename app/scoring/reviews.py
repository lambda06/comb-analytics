from app.models import Review, SignalResult
from app.scoring.confidence import review_confidence

def score_reviews(reviews: list[Review] | None) -> SignalResult:
    if reviews is None:
        return SignalResult(score=0.0, confidence=0.0, factors={})
        
    n_reviews = len(reviews)
    confidence = review_confidence(n_reviews)
    
    if n_reviews == 0:
        return SignalResult(score=50.0, confidence=confidence, factors={"n_reviews": 0})
        
    avg_rating = sum(r.rating for r in reviews) / n_reviews
    base_score = (avg_rating / 5.0) * 100.0
    
    # Negative skew penalty (1-2 star reviews)
    negative_reviews = sum(1 for r in reviews if r.rating <= 2)
    pct_negative = (negative_reviews / n_reviews) * 100.0
    
    # Penalty: 20 points if 100% negative, scaled
    penalty = (pct_negative / 100.0) * 20.0
    score = max(0.0, base_score - penalty)
    
    factors = {
        "n_reviews": n_reviews,
        "avg_rating": round(avg_rating, 2),
        "pct_negative": round(pct_negative, 1)
    }
    
    return SignalResult(score=round(score, 1), confidence=confidence, factors=factors)
