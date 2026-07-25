"""
FastAPI application — business health scoring API.

Endpoints
---------
GET /health                       → liveness probe
GET /businesses                   → list all business IDs + names
GET /score/all                    → score every business
GET /score/{business_id}          → score one business (no explanation)
GET /score/{business_id}/explain  → score + LLM narrative explanation
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.aggregator import score_business
from app.loader import get_business, load_businesses
from app.models import BusinessListItem, BusinessScoreResponse, HealthResponse


# ---------------------------------------------------------------------------
# Startup: validate data file is readable before we start accepting requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    businesses = load_businesses()          # raises if file is missing/malformed
    print(f"[startup] Loaded {len(businesses)} businesses from data file.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hive Business Health Scorer",
    description=(
        "Scores D2C businesses across four signal categories: revenue trend, "
        "customer reviews, retention & support, and ad efficiency. "
        "Uses confidence-weighted aggregation so missing signals abstain "
        "rather than penalise."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes — NOTE: /score/all MUST be declared before /score/{business_id}
# so FastAPI does not interpret "all" as a business_id path parameter.
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness probe — always returns 200 if the server is running."""
    return HealthResponse(status="ok")


@app.get("/businesses", response_model=list[BusinessListItem], tags=["meta"])
def list_businesses():
    """Return the list of all available businesses (id, name, category)."""
    return [
        BusinessListItem(
            business_id=b.business_id,
            business_name=b.business_name,
            category=b.category,
        )
        for b in load_businesses().values()
    ]


@app.get("/score/all", response_model=list[BusinessScoreResponse], tags=["scoring"])
def score_all():
    """
    Score every business in the dataset.
    Returns a list of BusinessScoreResponse objects sorted by composite_score descending.
    """
    results = [score_business(b) for b in load_businesses().values()]
    return sorted(results, key=lambda r: r.composite_score, reverse=True)


@app.get("/score/{business_id}", response_model=BusinessScoreResponse, tags=["scoring"])
def score_one(business_id: str):
    """
    Score a single business by ID (e.g. BIZ-006).
    Returns signal scores, composite, excluded signals, and confidence warning.
    No LLM explanation — use /score/{business_id}/explain for that.
    """
    biz = get_business(business_id)
    if biz is None:
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' not found.")
    return score_business(biz)


@app.get("/score/{business_id}/explain", response_model=BusinessScoreResponse, tags=["scoring"])
def score_and_explain(business_id: str):
    """
    Score a single business AND generate a Gemini-powered narrative explanation.
    The explanation cites only the computed signal factors — no hallucinations.
    Requires GEMINI_API_KEY to be set in the environment (.env file).
    """
    biz = get_business(business_id)
    if biz is None:
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' not found.")

    result = score_business(biz)

    # LLM explainer is imported lazily so the API remains usable even if
    # the Gemini key is not configured (score endpoints still work).
    try:
        from app.llm_explainer import generate_explanation
        result.explanation = generate_explanation(result)
    except Exception as exc:
        # Surface the error in the explanation field rather than crashing
        result.explanation = f"[LLM unavailable: {exc}]"

    return result
