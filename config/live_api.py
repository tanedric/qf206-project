"""
API key placeholders for the live-compatible ingestion pipeline.

Environment variables take precedence over these placeholders:
- POLYGON_API_KEY
- FINNHUB_API_KEY
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

POLYGON_API_KEY = "lIkF8EGitOC7684WjbEHDDJayCmp_pMg"
FINNHUB_API_KEY = "d736eg9r01qn7f07pccgd736eg9r01qn7f07pcd0"


def _clean_key(value: str | None) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.upper() == "INSERT_HERE":
        return None
    return cleaned


def resolve_api_keys() -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve API keys from environment first, then the placeholder module constants.
    """

    polygon = _clean_key(os.getenv("POLYGON_API_KEY")) or _clean_key(POLYGON_API_KEY)
    finnhub = _clean_key(os.getenv("FINNHUB_API_KEY")) or _clean_key(FINNHUB_API_KEY)
    return polygon, finnhub
