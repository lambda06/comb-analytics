# Hive Business Health Scorer

A FastAPI service that scores D2C businesses across four signal categories — revenue trend, customer reviews, retention & support, and ad efficiency — and generates a grounded LLM narrative using the Gemini API.

---

## Setup

**Prerequisites:** Python 3.11+

```bash
# 1. Clone / enter the project directory
cd "d:\Projects\AI\Hive Assignment"

# 2. Create and activate the virtual environment
python -m venv .venv
.venv\Scripts\activate     

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Gemini API key
copy .env.example .env
```

---

## Running the API

```bash
.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

---

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/businesses` | List all business IDs, names, and categories |
| `GET` | `/score/all` | Score all businesses, sorted by composite score descending |
| `GET` | `/score/{business_id}` | Score a single business (no LLM call) |
| `GET` | `/score/{business_id}/explain` | Score + Gemini narrative explanation |

> **Note:** The `/explain` endpoint requires `GEMINI_API_KEY` in `.env`. All scoring endpoints work without it.

---

## Architecture

```
app/
├── main.py              # FastAPI app — 5 routes
├── loader.py            # Loads + caches business_signals.json (lru_cache)
├── models.py            # Pydantic schemas: input data + API response shapes
├── aggregator.py        # Combines 4 signal scores into a composite
├── llm_explainer.py     # Builds grounded Gemini prompt + calls the API
└── scoring/
    ├── confidence.py    # Shared confidence curves (time-series, reviews, retention sub-signals)
    ├── revenue.py       # Revenue trend scorer
    ├── reviews.py       # Customer review scorer
    ├── retention.py     # Retention & support compound scorer
    └── ad_efficiency.py # Ad spend / ROAS efficiency scorer
data/
└── business_signals.json
```

**Key principle:** every module has a single responsibility. Adding a 5th signal means creating one new file in `scoring/`, calling it from `aggregator.py`, and updating the LLM prompt's `_SCORE_DRIVER` dict — nothing else changes.

---

## Scoring Design

### 1. Revenue Scorer

**Signal:** `revenue_by_month`

Computes Month-over-Month (MoM) percentage deltas and applies a **linear decay weighting** before averaging. The most recent month gets weight 1.0, decaying by 0.1 per month (floor 0.1). This ensures a business that declined sharply last month is penalised more than one whose decline happened 10 months ago — without needing explicit outlier detection.

```
weight(month) = max(0.1,  1.0 - 0.1 × months_ago)
weighted_avg  = Σ(delta × weight) / Σ(weight)
```

The weighted average is then mapped to a 0–100 score via a **sigmoid function** (`k=0.12`):

```
score = 100 / (1 + e^(-0.12 × weighted_avg))
```


---

### 2. Reviews Scorer

**Signal:** `customer_reviews`

```
base_score    = (avg_rating / 5.0) × 100
penalty       = (pct_negative_reviews / 100) × 20
score         = base_score - penalty
```

- `null` reviews → signal excluded (confidence 0)
- Empty `[]` reviews → neutral score 50, near-zero confidence (0.05)

Confidence by review count: 0 → 0.05, 1–3 → 0.25, 4–7 → 0.55, 8+ → 0.90.

---

### 3. Retention & Support Scorer (Compound)

**Signal:** `repeat_purchase_rate` + `customer_support_tickets`

The two sub-signals share a single "vote" in the final composite (preventing retention data from getting 2× the weight of any other category). Internally they are combined using **effective weights**:

```
effective_weight  = nominal_weight × sub_confidence
signal_confidence = Σ(nominal_weight × sub_confidence)   # 0 to 1
signal_score      = Σ(sub_score × eff_weight) / Σ(eff_weights)
```

### 4. Ad Efficiency Scorer

**Signal:** `ad_spend_by_month` × `revenue_by_month` (overlapping months)

```
ROAS(month) = revenue / ad_spend
avg_roas    = mean of all overlapping months
score       = min(avg_roas / 20, 1.0) × 100
```

Trend adjustment: if `latest_roas < avg_roas × 0.80` → −8 points (declining). If `latest_roas > avg_roas × 1.20` → +5 points (improving).

---

### 5. Aggregator

```
composite = Σ(score_i × confidence_i) / Σ(confidence_i)
```

Signals with `confidence = 0` are excluded from **both** the numerator and denominator. They abstain rather than score zero or inflate. This means:

- A business is never penalised for not having reviews
- A business with only 2 months of revenue data gets scored, but its revenue signal only weakly influences the composite (confidence 0.25)
- `low_confidence_warning: true` fires if any included signal has confidence < 0.35

---

### 6. LLM Explainer (Gemini)

The prompt is built programmatically from the computed `factors` dict, not from raw data. Each signal's primary score driver is explicitly tagged `[SCORE DRIVER]` so the LLM cites the right number. Internal scoring parameters (`rpr_sub_confidence`, etc.) are stripped before the prompt is built.

**Guardrails in the prompt:**
- Explicitly forbidden from mentioning metrics not in the data (CAC, churn, LTV, etc.)
- Confidence labels are explained in plain English to prevent the LLM contradicting them
- Required to identify the strongest and weakest signals, not just list numbers
- Required to note notable context (e.g. latest_revenue == peak_revenue)

### Signal Improvements
- **Revenue:** Incorporate absolute revenue level, not just growth rate. A business doing $10M/month declining 5% is very different from a business doing $50K/month declining 5%
- **Reviews:** Sentiment analysis on review text (not just star rating) using Gemini's embedding or classification API
- **Ad Efficiency:** Add a spend-trajectory component — a business cutting spend to inflate ROAS is not the same as one maintaining ROAS with stable spend
- **5th Signal:** Social proof / brand velocity (social mentions, search trend, referral rate) — the architecture supports this as a single new file in `scoring/`

### Infrastructure
- **Caching:** Cache `/explain` responses in Redis with a TTL so identical requests don't burn Gemini tokens
- **Async endpoints:** The Gemini call in `/explain` is synchronous; convert to `async def` with `asyncio` for better concurrency under load
- **Structured logging:** Replace `print()` with `structlog` for JSON log output
- **Config management:** Replace `os.getenv()` with a `pydantic-settings` `Settings` class for type-safe config validation at startup
- **Data versioning:** Business signals could be stored in a database (Postgres) rather than a static JSON file, enabling historical scoring and drift detection

### LLM Reliability
- **Output validation:** Parse and validate the LLM's explanation with a lightweight check (e.g. assert all `[SCORE DRIVER]` values appear in the text) before returning to the client
- **Retry logic:** Add exponential backoff on Gemini API timeouts
- **Prompt versioning:** Track prompt versions so explanation quality changes are auditable
