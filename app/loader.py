"""
Loads and validates business_signals.json into typed Business objects.
Returns a dict keyed by business_id for O(1) lookup.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.models import Business, BusinessData

DATA_PATH = Path(__file__).parent.parent / "data" / "business_signals.json"


@lru_cache(maxsize=1)
def load_businesses() -> dict[str, Business]:
    """
    Load and cache all businesses from disk.
    lru_cache ensures the file is read once per process lifetime.
    """
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    data = BusinessData(**raw)
    return {b.business_id: b for b in data.businesses}


def get_business(business_id: str) -> Business | None:
    """Return a single Business by ID, or None if not found."""
    return load_businesses().get(business_id)
