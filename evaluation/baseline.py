"""
DriftShield Keyword Baseline.

Provides a simple keyword-matching baseline model to compare classifier performance.
Used to compute baseline metrics and statistical significance.
"""

import re
from typing import Dict, Any, Optional

OUTDATED_PATTERNS: Dict[str, list] = {
    "cardiology": [
        "daily aspirin", "baby aspirin", "aspirin for everyone", "aspirin over 50",
        "aspirin over 60", "aspirin prevent heart", "statin only for high cholesterol",
        "130/80 is normal blood pressure",
    ],
    "diabetes": [
        "strict HbA1c", "hba1c below 6.5", "hba1c under 6.5", "6.5 is gold standard",
        "hba1c 6.5 for everyone", "metformin is not first line",
        "insulin is first line for type 2",
    ],
    "oncology": [
        "chemotherapy first line for nsclc", "chemo before immunotherapy",
        "psa screening for all men", "routine psa", "platinum first for lung",
    ],
    "neurology": [
        "tpa within 3 hours", "3 hour window stroke", "thrombectomy not helpful",
    ],
    "infectious_disease": [
        "antibiotics for bronchitis", "antibiotic for chest cold",
        "z-pack for bronchitis", "antibiotic for cough",
    ],
}

def keyword_baseline_predict(query: str) -> Dict[str, Any]:
    """Predicts presence of outdated clinical premise via keyword matching.

    Args:
        query: Clinical input query string.

    Returns:
        Dict containing label (1/0), verdict, matched pattern, and domain.
    """
    query_lower = query.lower()
    for domain, patterns in OUTDATED_PATTERNS.items():
        for pattern in patterns:
            if re.search(re.escape(pattern), query_lower):
                return {
                    "label": 1,
                    "verdict": "RISKY",
                    "matched_pattern": pattern,
                    "domain": domain
                }
    return {
        "label": 0,
        "verdict": "SAFE",
        "matched_pattern": None,
        "domain": None
    }
