from app.models import SupportTickets, SignalResult
from app.scoring.confidence import rpr_sub_confidence, support_sub_confidence

NOMINAL_RPR_W = 0.60
NOMINAL_SUP_W = 0.40

def score_retention_and_support(rpr: float | None, tickets: SupportTickets | None) -> SignalResult:
    rpr_conf = rpr_sub_confidence(rpr)
    
    total_tickets = tickets.total if tickets else None
    sup_conf = support_sub_confidence(total_tickets)
    
    eff_rpr_w = NOMINAL_RPR_W * rpr_conf
    eff_sup_w = NOMINAL_SUP_W * sup_conf
    total_eff_w = eff_rpr_w + eff_sup_w
    
    signal_confidence = total_eff_w
    
    if total_eff_w == 0.0:
        return SignalResult(score=0.0, confidence=0.0, factors={})
        
    # RPR score
    rpr_score_val = 0.0
    if rpr is not None:
        rpr_score_val = min(rpr / 0.50, 1.0) * 100.0
        
    # Support score
    support_score_val = 50.0
    if tickets is not None and tickets.total > 0:
        res_rate = (tickets.resolved_within_48h / tickets.total) * 100.0
        esc_rate = (tickets.escalated / tickets.total) * 100.0
        support_score_val = max(0.0, res_rate - (esc_rate * 0.5))
        
    # Compound score
    score = (rpr_score_val * eff_rpr_w + support_score_val * eff_sup_w) / total_eff_w
    
    # Only expose business-meaningful metrics in factors.
    # rpr_sub_confidence / support_sub_confidence are internal scoring machinery
    # — the signal-level confidence already communicates data quality to consumers.
    factors: dict = {}
    if rpr is not None:
        factors["rpr"] = rpr
        factors["rpr_score"] = round(rpr_score_val, 1)
    if tickets is not None:
        factors["total_tickets"] = tickets.total
        if tickets.total > 0:
            factors["resolution_rate"] = round((tickets.resolved_within_48h / tickets.total) * 100.0, 1)
            factors["escalation_rate"] = round((tickets.escalated / tickets.total) * 100.0, 1)
            
    return SignalResult(score=round(score, 1), confidence=round(signal_confidence, 2), factors=factors)
