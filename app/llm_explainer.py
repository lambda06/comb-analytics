"""
LLM Explainer — generates a grounded narrative explanation for a business score
using the Gemini API.
"""

from __future__ import annotations

import os

import google.generativeai as genai
from dotenv import load_dotenv

from app.models import BusinessScoreResponse

load_dotenv()

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def _build_prompt(result: BusinessScoreResponse) -> str:
    """
    Build a structured, grounded prompt from computed signal factors.

    Key design rules:
    - The score-driving metric is explicitly labelled for each signal so the
      LLM knows which number to cite (avoids citing a secondary metric).
    - The confidence label is explained in plain English so the LLM does not
      contradict it.
    """

    # Which factor key is the primary score driver for each signal
    _SCORE_DRIVER = {
        "revenue": "weighted_mom_growth",
        "reviews": "avg_rating",
        "retention_support": "rpr",
        "ad_efficiency": "avg_roas",
    }

    lines = [
        "You are a senior business analyst writing a concise health report for a D2C portfolio company.",
        "Do NOT contradict the confidence labels — they already reflect data volume.",
        "",
        f"Business        : {result.business_name} ({result.category})",
        f"Composite Score : {result.composite_score} / 100",
        "",
        "Signal Breakdown (the number marked [SCORE DRIVER] is what produced the score):",
    ]

    for signal_name, s in result.signal_scores.items():
        conf_label = (
            "high (large dataset — trust this signal)"
            if s.confidence >= 0.75
            else "medium (moderate dataset — treat with some caution)"
            if s.confidence >= 0.40
            else "low (thin dataset — interpret carefully)"
        )
        lines.append(
            f"\n  {signal_name.replace('_', ' ').title()} : {s.score}/100  "
            f"(confidence: {conf_label})"
        )
        driver_key = _SCORE_DRIVER.get(signal_name)
        for k, v in s.factors.items():
            tag = "  [SCORE DRIVER]" if k == driver_key else ""
            lines.append(f"      {k}: {v}{tag}")

    if result.excluded_signals:
        lines.append("")
        lines.append(
            f"  Signals excluded due to zero data: {', '.join(result.excluded_signals)}"
        )

    if result.low_confidence_warning:
        lines.append("")
        lines.append(
            "  NOTE: One or more signals have low confidence (thin data volume). "
            "Mention this when interpreting the composite score."
        )

    lines += [
        "",
        "Write 4–5 sentences with genuine analytical insight — NOT just a readout of the numbers.",
        "Your response MUST:",
        "  1. State which signal is the strongest and which is the weakest drag on the composite.",
        "  2. Cite the [SCORE DRIVER] for each signal (include % units where applicable).",
        "  3. Explain what the current composite score tells the investment team in plain terms.",
        "  4. Reference any notable context visible in the factors.",
        "  5. Do NOT cite any metric not listed above. Do NOT invent data gaps.",
        "Tone: direct and analytical, written for an internal investment team.",
    ]

    return "\n".join(lines)



def generate_explanation(result: BusinessScoreResponse) -> str:
    """
    Call Gemini with the structured prompt and return the explanation text.
    Raises if GEMINI_API_KEY is not set or the API call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_GEMINI_MODEL)

    prompt = _build_prompt(result)
    response = model.generate_content(prompt)
    return response.text.strip()
