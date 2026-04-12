"""
Polygon/Massive market-data adapter for the live-compatible pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


@dataclass
class PolygonMarketAdapter:
    api_key: str
    base_url: str = "https://api.polygon.io"
    timeout_seconds: int = 30
    min_interval_seconds: float = 0.0
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        self._last_request_time = 0.0

    def _respect_min_interval(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        remaining = float(self.min_interval_seconds) - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _retry_after_seconds(exc: HTTPError) -> Optional[float]:
        retry_after = exc.headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            return None

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}?{urlencode(query, doseq=True)}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                self._respect_min_interval()
                with urlopen(url, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._last_request_time = time.monotonic()
                if isinstance(payload, dict) and payload.get("status") == "ERROR":
                    raise RuntimeError(f"Polygon request failed for {path}: {payload}")
                return payload
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    if exc.code == 429:
                        raise RuntimeError(
                            "Polygon returned HTTP 429 (Too Many Requests). "
                            "Reduce the ticker count, increase request pacing, or retry later."
                        ) from exc
                    raise
                retry_after = self._retry_after_seconds(exc)
                sleep_for = (
                    retry_after
                    if retry_after is not None
                    else float(self.retry_backoff_seconds) * (2**attempt)
                )
                time.sleep(max(sleep_for, 0.0))
            except URLError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Polygon request failed for {path}: {exc}") from exc
                time.sleep(float(self.retry_backoff_seconds) * (2**attempt))

        raise RuntimeError(f"Polygon request failed for {path}: {last_error}")

    def fetch_daily_bars(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily aggregates for one ticker.

        Returns columns:
        - ticker
        - date
        - open, high, low, close
        - volume, vwap, transactions
        """

        payload = self._get_json(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date.isoformat()}/{end_date.isoformat()}",
            params={
                "adjusted": str(bool(adjusted)).lower(),
                "sort": "asc",
                "limit": 50_000,
            },
        )
        rows = payload.get("results", []) or []
        if not rows:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "transactions",
                ]
            )

        df = pd.DataFrame(rows).rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "transactions",
                "t": "timestamp",
            }
        )
        df["ticker"] = ticker.upper()
        df["date"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )
        for col in ("open", "high", "low", "close", "volume", "vwap", "transactions"):
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[
            [
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "transactions",
            ]
        ].sort_values("date", kind="mergesort").reset_index(drop=True)
