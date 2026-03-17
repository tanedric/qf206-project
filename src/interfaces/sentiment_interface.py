"""
Sentiment module interface (Phase 1 scaffold; DO NOT implement sentiment).

Another teammate will provide a sentiment analysis module later. This file
defines the expected contract so the rest of the system can integrate it
cleanly without depending on the implementation details.

## Optional input contract: Sentiment input (panel format)

Expected as a pandas.DataFrame named `sentiment_df`.

### Required columns (minimum)
- **date**: datetime64[ns]; timestamp aligned to the feature date
- **asset_id**: str/int; same identifier used throughout the pipeline
- **sentiment_score**: float; standardized sentiment signal (higher = more positive)

### Optional columns
- **sentiment_source**: str; e.g., "news", "social", "analyst_reports"
- **sentiment_confidence**: float; [0, 1] or similar
- **raw_text_count**: int; number of documents aggregated

### Key invariants
- One row per (date, asset_id) OR multiple rows allowed if later aggregated;
  if multiple rows are provided, the sentiment module should define how it
  aggregates to a single score per (date, asset_id) before merge.

## Integration expectation
- The core pipeline should treat sentiment as an optional join on
  (date, asset_id) that may add one or more features to `model_df`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SentimentProvider(Protocol):
    """
    Minimal interface for supplying sentiment features.

    The implementation may read from files/APIs and return `sentiment_df`
    matching this module's docstring contract.
    """

    def get_sentiment_panel(self, *args, **kwargs):
        """Return `sentiment_df` with columns: date, asset_id, sentiment_score."""

        raise NotImplementedError

