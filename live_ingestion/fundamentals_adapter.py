"""
Finnhub fundamentals adapter for the live-compatible pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

_FIELD_CONCEPTS: Dict[str, Tuple[str, ...]] = {
    "revenue": (
        "revenuefromcontractwithcustomerexcludingassessedtax",
        "salesrevenuenet",
        "revenue",
        "totalrevenue",
    ),
    "cogs": (
        "costofrevenue",
        "costofgoodssold",
        "costofsales",
    ),
    "sga": (
        "sellinggeneralandadministrativeexpense",
        "sellinggeneraladministrativeexpense",
    ),
    "interest_expense": (
        "interestexpenseanddebtexpense",
        "interestexpense",
    ),
    "operating_income": (
        "operatingincomeloss",
    ),
    "equity": (
        "stockholdersequity",
        "stockholdersequityincludingportionattributabletononcontrollinginterest",
        "totalstockholdersequity",
        "totalequity",
        "equity",
    ),
    "inventory": (
        "inventorynet",
        "inventoriesnetofreserves",
        "inventories",
    ),
    "ppe": (
        "propertyplantandequipmentnet",
        "propertyplantandequipmentandfinanceleaserightofuseassetafteraccumulateddepreciationandamortization",
    ),
    "assets": (
        "assets",
    ),
    "rnd": (
        "researchanddevelopmentexpense",
        "researchanddevelopmentexpenseexcludingacquiredinprocesscost",
        "researchdevelopmentandrelatedexpenses",
    ),
    "net_income": (
        "netincomeloss",
        "profitloss",
    ),
    "shares_outstanding": (
        "commonstocksharesoutstanding",
        "entitycommonstocksharesoutstanding",
        "weightedaveragenumberofdilutedsharesoutstanding",
        "weightedaveragenumberofsharesoutstandingdiluted",
        "weightedaveragenumberofsharesoutstandingbasicanddiluted",
        "weightedaveragenumberofsharesexcludingdilutiveeffectofequityinstruments",
    ),
}


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _flatten_concepts(node: Any) -> List[Tuple[str, float]]:
    found: List[Tuple[str, float]] = []
    if isinstance(node, dict):
        if "concept" in node and "value" in node:
            concept = str(node.get("concept", "")).strip().lower()
            value = _coerce_number(node.get("value"))
            if concept and value is not None:
                found.append((concept, value))
        for child in node.values():
            found.extend(_flatten_concepts(child))
    elif isinstance(node, list):
        for item in node:
            found.extend(_flatten_concepts(item))
    return found


def _parse_date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


@dataclass
class FinnhubFundamentalsAdapter:
    api_key: str
    base_url: str = "https://finnhub.io/api/v1"
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
        query["token"] = self.api_key
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}?{urlencode(query, doseq=True)}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                self._respect_min_interval()
                with urlopen(url, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._last_request_time = time.monotonic()
                if isinstance(payload, dict) and payload.get("error"):
                    raise RuntimeError(f"Finnhub request failed for {path}: {payload['error']}")
                return payload
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    if exc.code == 429:
                        raise RuntimeError(
                            "Finnhub returned HTTP 429 (Too Many Requests). "
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
                    raise RuntimeError(f"Finnhub request failed for {path}: {exc}") from exc
                time.sleep(float(self.retry_backoff_seconds) * (2**attempt))

        raise RuntimeError(f"Finnhub request failed for {path}: {last_error}")

    def fetch_company_profile(self, ticker: str) -> Dict[str, Any]:
        payload = self._get_json("/stock/profile2", params={"symbol": ticker.upper()})
        return payload if isinstance(payload, dict) else {}

    def fetch_financials_reported(self, ticker: str, *, freq: str) -> pd.DataFrame:
        """
        Fetch quarterly or annual reported financials and flatten useful fields.
        """

        payload = self._get_json(
            "/stock/financials-reported",
            params={"symbol": ticker.upper(), "freq": freq},
        )
        filings = payload.get("data", []) if isinstance(payload, dict) else []
        rows: List[Dict[str, Any]] = []
        for filing in filings or []:
            report = filing.get("report", {})
            concept_rows = _flatten_concepts(report)
            concept_map: Dict[str, float] = {}
            for concept, value in concept_rows:
                concept_map.setdefault(concept, value)

            parsed: Dict[str, Any] = {
                "ticker": ticker.upper(),
                "freq": freq,
                "year": filing.get("year"),
                "quarter": filing.get("quarter"),
                "form": filing.get("form"),
                "start_date": _parse_date(filing.get("startDate")),
                "report_end_date": _parse_date(filing.get("endDate")),
                "filed_date": _parse_date(filing.get("filedDate")),
                "accepted_date": _parse_date(filing.get("acceptedDate")),
            }
            available_date = parsed["accepted_date"]
            if pd.isna(available_date):
                available_date = parsed["filed_date"]
            if pd.isna(available_date):
                available_date = parsed["report_end_date"]
            parsed["available_date"] = available_date

            for field_name, concepts in _FIELD_CONCEPTS.items():
                parsed[field_name] = next(
                    (concept_map[c] for c in concepts if c in concept_map),
                    None,
                )
            rows.append(parsed)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "freq",
                    "year",
                    "quarter",
                    "form",
                    "start_date",
                    "report_end_date",
                    "filed_date",
                    "accepted_date",
                    "available_date",
                    *list(_FIELD_CONCEPTS),
                ]
            )

        df = pd.DataFrame(rows)
        numeric_cols = list(_FIELD_CONCEPTS)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return (
            df.sort_values(["available_date", "report_end_date"], kind="mergesort")
            .drop_duplicates(subset=["report_end_date"], keep="last")
            .reset_index(drop=True)
        )
