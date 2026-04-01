import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import argparse
from pypfopt import black_litterman, risk_models, expected_returns, EfficientFrontier
from pypfopt import objective_functions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import logging, contextlib, io
import traceback
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os
import sys
import json
import torch
import torch.nn as nn
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from sp500_wikipedia_universe import get_sp500_universe_for_date, ensure_membership_csv  # type: ignore[import]
from data_pipeline import compute_indicators, get_seqs, z_score_norm  # type: ignore[import]


# ─────────────────────────────────────────────
# WARNING FILTERS
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
API_KEY            = os.getenv("STOCKNEWS_API_KEY")

START_DATE         = "2015-01-01"
END_DATE           = "2026-01-01"
REBALANCE_FREQ     = "ME"

# S&P 500 universe filter – matched case-insensitively against GICS Sector and
# GICS Sub-Industry columns.  Set to None to use the full S&P 500.
SP500_GICS_FILTER  = "Semiconductor"
NEWS_LOOKBACK_DAYS = 15
SENTIMENT_SCALING  = 0.3
# Regime switch thresholds (average sentiment score: Positive=1, Neutral=0, Negative=-1)
# score <  REGIME_BAD_THRESHOLD  → bad   (min-vol, return ≥ risk-free)
# score >  REGIME_GOOD_THRESHOLD → good  (max return within 1-SD vol band of max-Sharpe)
# otherwise                       → neutral (max Sharpe)
REGIME_BAD_THRESHOLD  = -0.1
REGIME_GOOD_THRESHOLD =  0.1
RISK_FREE_RATE     = 0.02
CVAR_ALPHA         = 0.05
MIN_HISTORY_DAYS   = 1

# ── STRATEGY TOGGLES ──────────────────────────
ENABLE_MVO              = True
ENABLE_BLACK_LITTERMAN  = True
ENABLE_EQUAL_WEIGHT     = True
ENABLE_LSTM_MVO         = True
ENABLE_LSTM_BL          = True

# ── TRANSACTION COSTS ─────────────────────────
ENABLE_TRANSACTION_COSTS = True
TRANSACTION_COST_BPS     = 10

# ─────────────────────────────────────────────
# GLOBAL PLOT SETTINGS
# Keep DPI and figure sizes modest to stay under the 65536-pixel limit.
# Rule of thumb:  width_inches × savefig_dpi  <  65536
# e.g. 36 inches × 150 dpi = 5 400 px  ✓
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":           100,
    "figure.facecolor":     "white",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.labelsize":       11,
    "xtick.labelsize":      8,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "xtick.major.pad":      6,
    "xtick.minor.pad":      4,
    "xtick.major.size":     5,
    "xtick.minor.size":     3,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "axes.grid.axis":       "y",
    "grid.alpha":           0.35,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.6,
    "lines.linewidth":      2.0,
    "lines.antialiased":    True,
    "legend.framealpha":    0.85,
    "legend.edgecolor":     "lightgrey",
    "legend.loc":           "best",
    # Safe save settings – keep pixel count under 65536 per axis
    "savefig.dpi":          150,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
})

# ─────────────────────────────────────────────
# PER-FIGURE SIZE CONSTANTS
# Width chosen so that  width × 150 dpi  stays well below 65 536 px.
# 36 × 150 = 5 400 px  ← safe
# ─────────────────────────────────────────────
FIG_WIDE   = (36, 7)     # equity, drawdown, rolling metrics  (2× original width)
FIG_SQUARE = (24, 10)    # summary bar chart (2×3 grid)
FIG_WEIGHT = (36, 7)     # per-strategy weight chart (height multiplied by n_strategies)

# ─────────────────────────────────────────────
# STRATEGY & COST TOGGLES
# ─────────────────────────────────────────────
ENABLE_STRATEGIES = {
    "MVO":               ENABLE_MVO,
    "Black-Litterman":   ENABLE_BLACK_LITTERMAN,
    "Equal_Weight":      ENABLE_EQUAL_WEIGHT,
    "LSTM_MVO":          ENABLE_LSTM_MVO,
    "LSTM_BL":           ENABLE_LSTM_BL,
}
STRATEGIES = [s for s, enabled in ENABLE_STRATEGIES.items() if enabled]

TRANSACTION_COST = TRANSACTION_COST_BPS / 10_000

# ─────────────────────────────────────────────
# SILENT DOWNLOAD HELPER
# ─────────────────────────────────────────────
def silent_download(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    log  = logging.getLogger("yfinance")
    prev = log.level
    log.setLevel(logging.CRITICAL)
    buf  = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        raw = yf.download(tickers, start=start, end=end,
                          progress=False, auto_adjust=True)
    log.setLevel(prev)
    return raw

# ─────────────────────────────────────────────
# TICKER CLASSIFICATION
# ─────────────────────────────────────────────
def classify_tickers(tickers, start_date, end_date):
    active, delisted, ipo_map = [], [], {}
    start_dt = pd.Timestamp(start_date).tz_localize(None)
    for t in tickers:
        try:
            df = silent_download(t, start_date, end_date)
            if df.empty or len(df) < 5:
                print(f"  ✗  {t}: no data → delisted")
                delisted.append(t)
            elif df.index[0].tz_localize(None) <= start_dt + pd.Timedelta(days=5):
                print(f"  ✓  {t}: full history from {df.index[0].date()}")
                active.append(t)
            else:
                first = df.index[0].tz_localize(None)
                print(f"  ◑  {t}: IPO/data starts {first.date()}")
                ipo_map[t] = first
        except Exception as e:
            print(f"  ✗  {t}: error ({e}) → delisted")
            delisted.append(t)
    return active, ipo_map, delisted

def main():
    global START_DATE, END_DATE, ENABLE_MVO, ENABLE_BLACK_LITTERMAN, \
           ENABLE_EQUAL_WEIGHT, ENABLE_LSTM_MVO, ENABLE_LSTM_BL

    # ─────────────────────────────────────────────
    # ARGUMENT PARSING
    # ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Run the S&P 500 sector backtest.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--start", default=START_DATE, metavar="YYYY-MM-DD",
                        help=f"Backtest start date (default: {START_DATE})")
    parser.add_argument("--end",   default=END_DATE,   metavar="YYYY-MM-DD",
                        help=f"Backtest end date   (default: {END_DATE})")
    parser.add_argument(
        "--enable-strategies", nargs="+",
        metavar="STRATEGY",
        help=(
            "Whitelist of strategies to enable. All others are disabled.\n"
            "Choices: MVO  Black-Litterman  Equal_Weight  LSTM_MVO  LSTM_BL\n"
            "Example: --enable-strategies MVO LSTM_MVO"
        ),
    )
    parser.add_argument(
        "--disable-strategies", nargs="+",
        metavar="STRATEGY",
        help=(
            "Strategies to disable (applied after --enable-strategies).\n"
            "Example: --disable-strategies Equal_Weight LSTM_BL"
        ),
    )
    parser.add_argument(
        "--clear-cache", action="store_true", default=False,
        help="Delete news and sentiment cache files before running (default: False)",
    )
    args = parser.parse_args()

    # Apply date overrides
    START_DATE = args.start
    END_DATE   = args.end

    # Apply strategy overrides
    _all_strategies = ["MVO", "Black-Litterman", "Equal_Weight", "LSTM_MVO", "LSTM_BL"]
    _strategy_flags = {
        "MVO":             ENABLE_MVO,
        "Black-Litterman": ENABLE_BLACK_LITTERMAN,
        "Equal_Weight":    ENABLE_EQUAL_WEIGHT,
        "LSTM_MVO":        ENABLE_LSTM_MVO,
        "LSTM_BL":         ENABLE_LSTM_BL,
    }
    if args.enable_strategies:
        unknown = set(args.enable_strategies) - set(_all_strategies)
        if unknown:
            parser.error(f"Unknown strategies in --enable-strategies: {unknown}")
        for s in _all_strategies:
            _strategy_flags[s] = s in args.enable_strategies
    if args.disable_strategies:
        unknown = set(args.disable_strategies) - set(_all_strategies)
        if unknown:
            parser.error(f"Unknown strategies in --disable-strategies: {unknown}")
        for s in args.disable_strategies:
            _strategy_flags[s] = False
    ENABLE_MVO             = _strategy_flags["MVO"]
    ENABLE_BLACK_LITTERMAN = _strategy_flags["Black-Litterman"]
    ENABLE_EQUAL_WEIGHT    = _strategy_flags["Equal_Weight"]
    ENABLE_LSTM_MVO        = _strategy_flags["LSTM_MVO"]
    ENABLE_LSTM_BL         = _strategy_flags["LSTM_BL"]

    # Clear cache if requested
    if args.clear_cache:
        _base = os.path.dirname(os.path.abspath(__file__))
        for _path in (
            os.path.join(_base, "data", "news_cache.csv"),
            os.path.join(_base, "data", "sentiment_cache.csv"),
        ):
            if os.path.exists(_path):
                os.remove(_path)
                print(f"  ✓ Cleared cache: {_path}")

    print(f"── Configuration ───────────────────────────────────────────────────")
    print(f"  Start date : {START_DATE}")
    print(f"  End date   : {END_DATE}")
    print(f"  Strategies : { [s for s, v in _strategy_flags.items() if v] }")
    print(f"  Clear cache: {args.clear_cache}\n")

    print("── S&P 500 membership CSV ──────────────────────────────────────────")
    from sp500_wikipedia_universe import _get_universe_from_frame, _parse_date  # type: ignore[import]
    _membership_df = ensure_membership_csv()
    print("✓ S&P 500 membership ready\n")

    # Collect every ticker that was ever in the filtered universe across the full
    # backtest window, not just at START_DATE.  This ensures tickers that join the
    # index mid-backtest are downloaded upfront and eligible when they become members.
    def _all_tickers_in_window(membership_df):
        start_ts = _parse_date(START_DATE)
        end_ts   = _parse_date(END_DATE)
        mask = (
            membership_df["start_date"].le(end_ts) &
            (membership_df["end_date"].isna() | membership_df["end_date"].ge(start_ts))
        )
        if SP500_GICS_FILTER:
            f = SP500_GICS_FILTER.lower()
            sector_mask = pd.Series(False, index=membership_df.index)
            for col in ("gics_sub_industry", "gics_sector"):
                if col in membership_df.columns:
                    sector_mask |= membership_df[col].astype(str).str.lower().str.contains(f, na=False)
            mask &= sector_mask
        return sorted(membership_df.loc[mask, "ticker"].dropna().unique().tolist())

    sp500_initial = _all_tickers_in_window(_membership_df)
    # If the filter returns nothing the CSV was likely built before sector columns
    # were added — force a rebuild and retry once.
    if SP500_GICS_FILTER and not sp500_initial:
        print("  ⚠ GICS filter returned 0 tickers — rebuilding membership CSV …")
        _membership_df = ensure_membership_csv(force_refresh=True)
        sp500_initial = _all_tickers_in_window(_membership_df)
    if not sp500_initial:
        raise RuntimeError(
            f"S&P 500 universe is empty between {START_DATE} and {END_DATE}"
            + (f" with GICS filter '{SP500_GICS_FILTER}'" if SP500_GICS_FILTER else "")
            + ". Check the filter string or date range."
        )
    filter_label = f"  (filter: '{SP500_GICS_FILTER}')" if SP500_GICS_FILTER else ""
    print(f"✓ S&P 500 ever-members {START_DATE}→{END_DATE}: {len(sp500_initial)} tickers{filter_label}\n")

    print("── Ticker Classification ───────────────────────────────────────")
    active_tickers, ipo_map, delisted_tickers = classify_tickers(
        sp500_initial, START_DATE, END_DATE)
    all_valid_tickers = active_tickers + list(ipo_map.keys())
    print(f"\n  Active   : {active_tickers}")
    print(f"  IPO      : { {k: v.date() for k, v in ipo_map.items()} }")
    print(f"  Delisted : {delisted_tickers}\n")

    # ─────────────────────────────────────────────
    # DATA DOWNLOAD
    # ─────────────────────────────────────────────
    raw    = silent_download(all_valid_tickers, START_DATE, END_DATE)
    prices = (raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"])
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.ffill()

    for ticker, ipo_date in ipo_map.items():
        if ticker in prices.columns:
            prices.loc[prices.index < ipo_date, ticker] = np.nan

    returns       = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    returns.index = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()

    print(f"✓ Price data ready  |  {len(prices)} rows  |  universe: {all_valid_tickers}")

    # ─────────────────────────────────────────────
    # REBALANCE DATES
    # ─────────────────────────────────────────────
    def snap_to_trading_day(dates, trading_index):
        snapped = []
        for d in pd.DatetimeIndex(dates).tz_localize(None).normalize():
            future = trading_index[trading_index >= d]
            if len(future) > 0:
                snapped.append(future[0])
        return pd.DatetimeIndex(sorted(set(snapped)))

    rebalance_dates = snap_to_trading_day(
        pd.date_range(start=START_DATE, end=END_DATE, freq=REBALANCE_FREQ),
        returns.index,
    )
    print(f"\n✓ Rebalance dates ({len(rebalance_dates)}):")
    for d in rebalance_dates:
        print(f"    {d.date()}")

    # ─────────────────────────────────────────────
    # NEWS & SENTIMENT
    # ─────────────────────────────────────────────
    NEWS_CACHE_PATH      = "./data/news_cache.csv"
    SENTIMENT_CACHE_PATH = "./data/sentiment_cache.csv"
    _NEWS_CACHE_COLS = ["window_start", "window_end", "title", "tickers", "sentiment", "date"]
    _SENTIMENT_CACHE_COLS = ["rebalance_date", "ticker", "view", "omega"]

    def _load_news_cache():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), NEWS_CACHE_PATH)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for col in ("window_start", "window_end"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).dt.normalize()
            return df
        return pd.DataFrame(columns=_NEWS_CACHE_COLS)

    def _save_news_cache(cache_df):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), NEWS_CACHE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cache_df.to_csv(path, index=False)

    def _load_sentiment_cache():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENTIMENT_CACHE_PATH)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["rebalance_date"] = pd.to_datetime(df["rebalance_date"]).dt.normalize()
            return df
        return pd.DataFrame(columns=_SENTIMENT_CACHE_COLS)

    def _save_sentiment_cache(cache_df):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENTIMENT_CACHE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cache_df.to_csv(path, index=False)

    def _get_cached_views(sentiment_cache, rebalance_date, tickers):
        """Return (views dict, omega_vec list) if all tickers cached for this date, else None."""
        date_ts = pd.Timestamp(rebalance_date).normalize()
        rows = sentiment_cache[sentiment_cache["rebalance_date"].eq(date_ts)]
        if rows.empty:
            return None
        cached_tickers = set(rows["ticker"].tolist())
        if not set(tickers).issubset(cached_tickers):
            return None
        rows = rows.set_index("ticker")
        views     = {t: float(rows.loc[t, "view"])  for t in tickers}
        omega_vec = [float(rows.loc[t, "omega"]) for t in tickers]
        return views, omega_vec

    def _store_views_in_cache(sentiment_cache, rebalance_date, views, omega_vec, tickers):
        date_ts = pd.Timestamp(rebalance_date).normalize()
        # Remove any existing rows for this date to avoid duplicates
        keep = ~sentiment_cache["rebalance_date"].eq(date_ts)
        sentiment_cache = sentiment_cache[keep].copy()
        new_rows = pd.DataFrame([
            {"rebalance_date": date_ts, "ticker": t,
             "view": views[t], "omega": omega_vec[i]}
            for i, t in enumerate(tickers)
        ])
        return pd.concat([sentiment_cache, new_rows], ignore_index=True)

    def _articles_from_cache(cache_df, start, end):
        """Return list-of-dicts for a window that was previously fetched."""
        mask = (
            cache_df["window_start"].eq(pd.Timestamp(start).normalize()) &
            cache_df["window_end"].eq(pd.Timestamp(end).normalize())
        )
        rows = cache_df.loc[mask]
        articles = []
        for _, row in rows.iterrows():
            tickers_val = row["tickers"]
            if isinstance(tickers_val, str):
                import ast
                try:
                    tickers_val = ast.literal_eval(tickers_val)
                except Exception:
                    tickers_val = [t.strip() for t in tickers_val.split(",")]
            articles.append({
                "title":     row.get("title"),
                "tickers":   tickers_val,
                "sentiment": row.get("sentiment"),
                "date":      row.get("date"),
            })
        return articles

    def fetch_news(tickers, start, end, max_pages=1):
        """Return articles for the window, using the local cache when available."""
        nonlocal _news_cache
        start_ts = pd.Timestamp(start).normalize()
        end_ts   = pd.Timestamp(end).normalize()

        # Check cache: if any rows exist for this exact window, use them
        window_cached = (
            _news_cache["window_start"].eq(start_ts) &
            _news_cache["window_end"].eq(end_ts)
        ).any()

        if window_cached:
            return _articles_from_cache(_news_cache, start_ts, end_ts)

        # Not cached — call the API
        articles = []
        for page in range(1, max_pages + 1):
            url = "https://stocknewsapi.com/api/v1"
            params = {
                "tickers": ",".join(tickers),
                "items":   50,
                "page":    page,
                "date":    f"{start_ts.strftime('%m%d%Y')}-{end_ts.strftime('%m%d%Y')}",
                "token":   API_KEY,
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                data = r.json()
                if "data" in data:
                    articles.extend(data["data"])
                if len(data.get("data", [])) == 0:
                    break
            except Exception as e:
                print(f"     [API] ERROR: {e}")
                break

        # Persist to cache (even if empty, so we don't re-query)
        new_rows = pd.DataFrame([
            {
                "window_start": start_ts,
                "window_end":   end_ts,
                "title":        a.get("title"),
                "tickers":      str(a.get("tickers", [])),
                "sentiment":    a.get("sentiment"),
                "date":         a.get("date"),
            }
            for a in articles
        ] or [{"window_start": start_ts, "window_end": end_ts,
               "title": None, "tickers": "[]", "sentiment": None, "date": None}])
        _news_cache = pd.concat([_news_cache, new_rows], ignore_index=True)
        _save_news_cache(_news_cache)

        return articles


    def sentiment_views(articles, tickers):
        score_map    = {"Positive": 1, "Neutral": 0, "Negative": -1}
        scores       = {t: [] for t in tickers}
        for art in articles:
            for t in tickers:
                if t in art.get("tickers", []):
                    scores[t].append(score_map.get(art.get("sentiment"), 0))

        views, omega_vec = {}, []
        for t in tickers:
            n = len(scores[t])
            if n == 0:
                views[t] = 0.0
                omega_vec.append(1e6)
            else:
                avg       = np.mean(scores[t])
                sem       = (np.std(scores[t]) / np.sqrt(n)) if n > 1 else 1.0
                views[t]  = avg * SENTIMENT_SCALING
                omega_vec.append(max(sem ** 2, 1e-6))
        return views, omega_vec


    def overall_sentiment_regime(articles, news_cache, window_start, window_end):
        """Compute overall sentiment regime from articles or cached news.

        Uses pre-fetched articles if non-empty, otherwise reads from the news
        cache for the same window.  Returns 'bad', 'neutral', or 'good'.
        """
        score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
        scores = []

        if articles:
            for a in articles:
                s = a.get("sentiment")
                if s in score_map:
                    scores.append(score_map[s])
        else:
            # Pull from cache for this exact window
            cached = _articles_from_cache(
                news_cache,
                pd.Timestamp(window_start).normalize(),
                pd.Timestamp(window_end).normalize(),
            )
            for a in cached:
                s = a.get("sentiment")
                if s in score_map:
                    scores.append(score_map[s])

        if not scores:
            avg = 0.0
        else:
            avg = float(np.mean(scores))

        if avg < REGIME_BAD_THRESHOLD:
            regime = "bad"
        elif avg > REGIME_GOOD_THRESHOLD:
            regime = "good"
        else:
            regime = "neutral"

        print(f"     Regime: {regime}  (avg_sentiment={avg:.3f}, n={len(scores)})")
        return regime

    def regime_weights(mu, S, regime):
        """Return weight array for live_tickers according to the sentiment regime.

        bad     → min-volatility with expected-return ≥ RISK_FREE_RATE constraint
        neutral → max Sharpe (standard)
        good    → max return subject to volatility ≤ vol_sharpe + 1*std(frontier vols)
        """
        n = len(mu)
        fallback = np.full(n, 1 / n)

        if regime == "neutral":
            return max_sharpe_weights(mu, S)

        if regime == "bad":
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef.add_constraint(lambda w: w @ mu - RISK_FREE_RATE)
                ef.min_volatility()
                w = np.array(list(ef.clean_weights().values()))
                w = np.clip(w, 0, 1)
                return w / w.sum() if w.sum() > 0 else fallback
            except Exception:
                try:
                    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                    ef.min_volatility()
                    w = np.array(list(ef.clean_weights().values()))
                    w = np.clip(w, 0, 1)
                    return w / w.sum() if w.sum() > 0 else fallback
                except Exception:
                    return fallback

        # regime == "good": max return within vol_sharpe + 1 std of frontier vols
        try:
            # Compute max-Sharpe vol as reference
            ef_sharpe = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef_sharpe.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef_sharpe.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            w_sharpe = np.array(list(ef_sharpe.clean_weights().values()))
            vol_sharpe = float(np.sqrt(w_sharpe @ S.values @ w_sharpe))

            # Sample a few frontier points to estimate spread of frontier vols
            frontier_vols = []
            for target in np.linspace(float(mu.min()), float(mu.max()), 20):
                try:
                    ef_f = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                    ef_f.efficient_return(target)
                    wf = np.array(list(ef_f.clean_weights().values()))
                    frontier_vols.append(float(np.sqrt(wf @ S.values @ wf)))
                except Exception:
                    pass
            vol_std = float(np.std(frontier_vols)) if len(frontier_vols) > 1 else vol_sharpe * 0.1
            vol_cap = vol_sharpe + vol_std

            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.add_constraint(lambda w: vol_cap ** 2 - w @ S.values @ w)
            ef.efficient_return(float(mu.max()))
            w = np.array(list(ef.clean_weights().values()))
            w = np.clip(w, 0, 1)
            return w / w.sum() if w.sum() > 0 else fallback
        except Exception:
            # Fall back to max Sharpe if the constrained problem fails
            return max_sharpe_weights(mu, S)

    def build_relative_views(views_dict, tickers):
        sorted_t       = sorted(views_dict, key=views_dict.get, reverse=True)
        P_rows, q_vals = [], []
        mid            = len(sorted_t) // 2
        for winner in sorted_t[:mid]:
            for loser in sorted_t[mid:]:
                diff = views_dict[winner] - views_dict[loser]
                if abs(diff) > 0.05:
                    row                        = np.zeros(len(tickers))
                    row[tickers.index(winner)] =  1
                    row[tickers.index(loser)]  = -1
                    P_rows.append(row)
                    q_vals.append(float(diff))
        if len(P_rows) == 0:
            return None, None
        P = np.array(P_rows, dtype=float)
        Q = np.array(q_vals,  dtype=float)
        return P, Q

    # ─────────────────────────────────────────────
    # REAL MARKET CAPS
    # ─────────────────────────────────────────────
    def fetch_market_caps(tickers):
        caps = {}
        for t in tickers:
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    info = yf.Ticker(t).info
                caps[t] = info.get("marketCap") or 1
            except Exception:
                caps[t] = 1
        return pd.Series(caps, dtype=float)

    # Only fetch market caps if Black-Litterman is enabled
    if ENABLE_STRATEGIES["Black-Litterman"]:
        print("\nFetching market caps …")
        MARKET_CAPS_FULL = fetch_market_caps(all_valid_tickers)
        print(f"  {MARKET_CAPS_FULL.to_dict()}\n")
    else:
        MARKET_CAPS_FULL = pd.Series({t: 1.0 for t in all_valid_tickers})
        print("\nMarket caps: skipped (Black-Litterman disabled)\n")

    # ─────────────────────────────────────────────
    # LSTM MODEL
    # ─────────────────────────────────────────────
    _MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    _MODEL_PATH  = os.path.join(_MODEL_DIR, "best_model.pt")
    _PARAM_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "param_dict.json")
    _LSTM_SEQ_LEN = 9
    _N_FEATURES   = 12

    class _LSTMModel(nn.Module):
        def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    _lstm_model = None
    _lstm_param_dict = None

    def _load_lstm():
        nonlocal _lstm_model, _lstm_param_dict
        if _lstm_model is not None:
            return
        checkpoint = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
        # Support both raw state-dict and checkpoint dicts
        state_dict = checkpoint.get("model_state_dict", checkpoint) \
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint \
            else checkpoint
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        hidden  = cfg.get("hidden_size",  64)
        layers  = cfg.get("num_layers",   2)
        dropout = cfg.get("dropout",      0.2)
        model = _LSTMModel(_N_FEATURES, hidden, layers, dropout)
        model.load_state_dict(state_dict)
        model.eval()
        _lstm_model = model
        if os.path.exists(_PARAM_PATH):
            with open(_PARAM_PATH) as f:
                raw = json.load(f)
            _lstm_param_dict = {k: tuple(v) for k, v in raw.items()}
        print("✓ LSTM model loaded\n")

    _indicator_cache: dict = {}  # keyed by (ticker, end_month_str) → pd.DataFrame

    def predict_lstm_returns(tickers, as_of_date):
        """Return a pd.Series of predicted monthly returns keyed by ticker.
        Tickers with insufficient history are omitted.
        Shared data (FF5, industry ETF, bulk prices) are fetched once per call."""
        import data_pipeline as dp  # type: ignore[import]
        _load_lstm()
        preds = {}
        end_str   = as_of_date.strftime("%Y-%m-%d")
        # Need enough history for longest lookback (36m) + seq_len months
        raw_start = (as_of_date - pd.DateOffset(years=5)).strftime("%Y-%m-%d")

        # ── Shared pre-fetches (once per rebalance) ────────────────────────────
        ff5              = dp.fetch_ff5_factors(raw_start, end_str)
        industry_returns = dp.fetch_industry_tracking_stock(raw_start, end_str)
        market_returns   = dp.fetch_market_returns(raw_start, end_str) if ff5 is None else None

        # Bulk price download for all tickers at once
        bulk_prices = silent_download(tickers, raw_start, end_str)

        for t in tickers:
            try:
                cache_key = (t, end_str)
                if cache_key in _indicator_cache:
                    df = _indicator_cache[cache_key]
                else:
                    # Extract this ticker's OHLCV from the bulk download
                    if isinstance(bulk_prices.columns, pd.MultiIndex):
                        try:
                            daily_data = bulk_prices.xs(t, axis=1, level=1)
                        except KeyError:
                            print(f"     LSTM skip {t}: not in bulk download")
                            continue
                    else:
                        daily_data = bulk_prices.copy()

                    if daily_data.empty or len(daily_data) < 5:
                        print(f"     LSTM skip {t}: insufficient price data")
                        continue

                    daily_data.index = pd.to_datetime(daily_data.index)
                    daily_data = daily_data.sort_index()

                    df = dp.compute_indicators(
                        t,
                        start_date=raw_start,
                        end_date=end_str,
                        daily_data=daily_data,
                        ff5=ff5,
                        industry_returns=industry_returns,
                        market_returns=market_returns,
                    )
                    _indicator_cache[cache_key] = df

                if _lstm_param_dict is not None:
                    dp.param_dict = _lstm_param_dict
                    feature_cols = [c for c in df.columns if c != "returns"]
                    df = df.copy()
                    df[feature_cols] = df[feature_cols].apply(dp.z_score_norm, axis=0)
                seqs, _ = dp.get_seqs(df, seq_len=_LSTM_SEQ_LEN)
                if not seqs:
                    continue
                x = torch.tensor(seqs[-1:], dtype=torch.float32)   # last sequence only
                with torch.no_grad():
                    pred = _lstm_model(x).item()
                preds[t] = pred
            except Exception as e:
                print(f"     LSTM skip {t}: {e}")
        return pd.Series(preds, dtype=float)

    # Only load LSTM if either LSTM strategy is enabled
    if ENABLE_STRATEGIES["LSTM_MVO"] or ENABLE_STRATEGIES["LSTM_BL"]:
        _load_lstm()

    # ─────────────────────────────────────────────
    # PORTFOLIO OPTIMISATION
    # ─────────────────────────────────────────────
    def max_sharpe_weights(mu, S, gamma_l2=0.1):
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg, gamma=gamma_l2)
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            w  = np.array(list(ef.clean_weights().values()))
            w  = np.clip(w, 0, 1)
            w  = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
            return w
        except Exception:
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                ef.min_volatility()
                w  = np.array(list(ef.clean_weights().values()))
                w  = np.clip(w, 0, 1)
                w  = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
                return w
            except Exception:
                n = len(mu)
                return np.full(n, 1 / n)


    def safe_weights(w_dict, tickers):
        arr = np.array([w_dict.get(t, 0.0) for t in tickers])
        arr = np.clip(arr, 0, 1)
        total = arr.sum()
        if total <= 0:
            arr = np.full(len(tickers), 1 / len(tickers))
        else:
            arr = arr / total
        return dict(zip(tickers, arr))

    # ─────────────────────────────────────────────
    # AXIS & LABEL HELPERS
    # ─────────────────────────────────────────────
    def monthly_xaxis(ax, rotation=60, fontsize=8):
        """Monthly major ticks; year shown only on January to avoid crowding."""
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        ax.grid(which="minor", axis="x", alpha=0.12, linestyle=":")
        ax.grid(which="major", axis="x", alpha=0.25)

        def _fmt(x, __pos):
            dt = mdates.num2date(x)
            return f"{dt.strftime('%b')}\n{dt.year}" if dt.month == 1 else dt.strftime("%b")

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
        plt.setp(ax.get_xticklabels(),
                 rotation=rotation, ha="center", fontsize=fontsize)


    def rebalance_xaxis(ax, dates, rotation=60, fontsize=8):
        """One tick per rebalance date; year shown only when it changes."""
        x_pos  = np.arange(len(dates))
        prev_year = None
        labels = []
        for d in dates:
            if d.year != prev_year:
                labels.append(f"{d.strftime('%b')}\n{d.year}")
                prev_year = d.year
            else:
                labels.append(d.strftime("%b"))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=rotation,
                           ha="center", fontsize=fontsize)


    def expand_xlim(ax, margin_days=10):
        left, right = ax.get_xlim()
        ax.set_xlim(left - margin_days, right + margin_days)


    def apply_layout(fig, extra_bottom=0.18):
        fig.tight_layout()
        fig.subplots_adjust(bottom=extra_bottom)

    # ─────────────────────────────────────────────
    # TRANSACTION COST HELPER
    # ─────────────────────────────────────────────
    def apply_transaction_costs(new_weights_dict, old_weights_dict,
                                 portfolio_value, all_tickers):
        if not ENABLE_TRANSACTION_COSTS:
            return portfolio_value
        total_cost = 0.0
        for t in all_tickers:
            old_w    = old_weights_dict.get(t, 0.0)
            new_w    = new_weights_dict.get(t, 0.0)
            turnover = abs(new_w - old_w)
            total_cost += turnover * TRANSACTION_COST * portfolio_value
        return portfolio_value - total_cost

    # ─────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────
    def compute_metrics(equity_series):
        years      = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
        total_ret  = equity_series.iloc[-1] / equity_series.iloc[0] - 1
        ann_ret    = (1 + total_ret) ** (1 / years) - 1
        daily_rets = equity_series.pct_change().dropna()
        ann_vol    = daily_rets.std() * np.sqrt(252)
        sharpe     = (ann_ret - RISK_FREE_RATE) / ann_vol
        drawdown   = equity_series / equity_series.cummax() - 1
        max_dd     = drawdown.min()
        calmar     = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
        var        = daily_rets.quantile(CVAR_ALPHA)
        cvar       = daily_rets[daily_rets <= var].mean()
        return {
            "Annual Return":     ann_ret,
            "Annual Volatility": ann_vol,
            "Sharpe Ratio":      sharpe,
            "Max Drawdown":      max_dd,
            "Calmar Ratio":      calmar,
            "CVaR (95%)":        cvar,
        }

    # ─────────────────────────────────────────────
    # AVAILABLE TICKERS HELPER
    # ─────────────────────────────────────────────
    def available_tickers(date, all_tickers, ipo_map, min_history=MIN_HISTORY_DAYS):
        result = []
        for t in all_tickers:
            if t in ipo_map:
                if returns.loc[:date, t].dropna().__len__() >= min_history:
                    result.append(t)
            else:
                result.append(t)
        return result

    # ─────────────────────────────────────────────
    # BACKTEST LOOP
    # BUG FIX: the rebalance block and the daily P&L block must both be
    # *inside* the `for i in range(...)` loop.  Previously the rebalance
    # block was accidentally dedented one level, so it only executed once
    # (on the very last value of `date`).
    # ─────────────────────────────────────────────
    weight_history = {s: [] for s in STRATEGIES}
    weight_dates   = []

    prev_weights   = {s: {t: 1 / len(all_valid_tickers) for t in all_valid_tickers}
                      for s in STRATEGIES}
    w              = {s: {t: 1 / len(all_valid_tickers) for t in all_valid_tickers}
                      for s in STRATEGIES}

    # ── Vectorised P&L arrays ─────────────────────────────────────────────────
    returns_arr = returns.values                        # shape (n_days, n_tickers)
    ticker_idx  = {t: i for i, t in enumerate(returns.columns)}
    n_tickers   = len(all_valid_tickers)
    valid_idx   = [ticker_idx[t] for t in all_valid_tickers]

    _init_w = 1 / n_tickers
    w_arr   = {s: np.full(n_tickers, _init_w) for s in STRATEGIES}

    # ── Covariance caching ────────────────────────────────────────────────────
    _prev_window_key = None
    _prev_S          = None
    _prev_mu         = None

    equity         = pd.DataFrame(100.0, index=returns.index, columns=STRATEGIES)
    news_log          = []
    _news_cache       = _load_news_cache()
    _sentiment_cache  = _load_sentiment_cache()

    print(f"\nStarting backtest …")
    print(f"  Strategies        : {STRATEGIES}")
    print(f"  Transaction costs : {'ON  (%s bps one-way)' % TRANSACTION_COST_BPS if ENABLE_TRANSACTION_COSTS else 'OFF'}\n")

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────────
    for i in range(1, len(returns)):
        date    = returns.index[i]

        # ── REBALANCE BLOCK ───────────────────────────────────────────────────────
        # Indented correctly inside the for-loop.
        if date in rebalance_dates:

            # Build candidate list from S&P 500 constituents at this rebalance date,
            # restricted to tickers we have price data for.
            sp500_at_date = set(_get_universe_from_frame(_membership_df, date, gics_filter=SP500_GICS_FILTER))
            av_tick = available_tickers(date, all_valid_tickers, ipo_map)
            av_tick = [t for t in av_tick if t in sp500_at_date]

        if date in rebalance_dates and len(av_tick) > 1:

            window_start = returns.index[max(0, i - 126)]
            window = (
                returns
                .loc[window_start:date, av_tick]
                .iloc[:-1]           # exclude today (not yet closed)
                .dropna(axis=1)
            )
            live_tickers = list(window.columns)
            n_live       = len(live_tickers)

            print(f"\n  → {date.date()}  window={len(window)}  live={n_live}")

            if len(window) >= MIN_HISTORY_DAYS and n_live >= 2:
                try:
                    # ── Initialise view variables ──────────────────────────────
                    P, Q      = None, None
                    views     = {t: 0.0 for t in live_tickers}
                    omega_vec = [1e6] * len(live_tickers)
                    articles  = []

                    _window_key = (frozenset(live_tickers), date)
                    if _window_key == _prev_window_key:
                        S  = _prev_S
                        mu = _prev_mu
                    else:
                        S  = risk_models.CovarianceShrinkage(
                                 window, returns_data=True).ledoit_wolf()
                        mu = expected_returns.ema_historical_return(
                                 window, returns_data=True, span=180)
                        _prev_S          = S
                        _prev_mu         = mu
                        _prev_window_key = _window_key

                    # ── News & sentiment (always fetched for regime switch) ────
                    articles   = []
                    _today     = pd.Timestamp.now().normalize()
                    news_end   = min(date, _today)
                    news_start = news_end - pd.Timedelta(days=NEWS_LOOKBACK_DAYS)
                    cached_sv  = _get_cached_views(_sentiment_cache, date, live_tickers)
                    if cached_sv is not None:
                        views, omega_vec = cached_sv
                    else:
                        articles = fetch_news(live_tickers, news_start, news_end)
                        news_log.extend({
                            "rebalance_date": date.date(),
                            "title":     a.get("title"),
                            "tickers":   a.get("tickers"),
                            "sentiment": a.get("sentiment"),
                            "date":      a.get("date"),
                        } for a in articles)
                        views, omega_vec = sentiment_views(articles, live_tickers)
                        _sentiment_cache = _store_views_in_cache(
                            _sentiment_cache, date, views, omega_vec, live_tickers)
                        _save_sentiment_cache(_sentiment_cache)

                    # ── Regime switch ──────────────────────────────────────────
                    regime = overall_sentiment_regime(
                        articles, _news_cache, news_start, news_end)

                    # ── MVO ───────────────────────────────────────────────────
                    if ENABLE_STRATEGIES["MVO"]:
                        w["MVO"] = dict(zip(live_tickers,
                                            regime_weights(mu, S, regime)))

                    P, Q = build_relative_views(views, live_tickers)

                    # ── Shared BL model builder ────────────────────────────────
                    def _build_bl_model(pi_returns, tau):
                        """Build a BlackLittermanModel using the shared sentiment views.
                        Returns None when no views are available (caller falls back to prior)."""
                        idx_map = {t: j for j, t in enumerate(live_tickers)}
                        print(f"     Views        : { {k: round(v,4) for k,v in views.items()} }")
                        print(f"     Relative P/Q : {P is not None} "
                              f"({'%d pairs' % len(Q) if Q is not None else 'none'})")
                        if P is not None and Q is not None:
                            view_omegas  = [np.mean([omega_vec[j]
                                            for j in [jj for jj, v in enumerate(row) if v != 0]])
                                            for row in P]
                            omega_matrix = np.diag(view_omegas).astype(float)
                            return black_litterman.BlackLittermanModel(
                                S, pi=pi_returns, P=P, Q=Q,
                                omega=omega_matrix, tau=tau)
                        active_views = {t: v for t, v in views.items() if v != 0}
                        if active_views:
                            active_omega = np.diag([
                                omega_vec[idx_map[t]] for t in active_views
                            ]).astype(float)
                            return black_litterman.BlackLittermanModel(
                                S, pi=pi_returns,
                                absolute_views=active_views,
                                omega=active_omega,
                                tau=tau)
                        print("     No views — falling back to prior directly")
                        return None

                    def _bl_weights(pi_returns, tau):
                        """Build BL model and optimise using the current sentiment regime.
                        Falls back to regime_weights on pi directly when no views exist."""
                        bl_model = _build_bl_model(pi_returns, tau)
                        if bl_model is None:
                            return regime_weights(pi_returns, S, regime)
                        ret_bl = bl_model.bl_returns()
                        if not np.all(np.isfinite(ret_bl)):
                            print("     ⚠ BL returns not finite — falling back to prior")
                            return regime_weights(pi_returns, S, regime)
                        return regime_weights(ret_bl, S, regime)

                    # ── Black-Litterman ───────────────────────────────────────
                    if ENABLE_STRATEGIES["Black-Litterman"]:
                        market_caps = MARKET_CAPS_FULL.reindex(live_tickers).fillna(1)
                        pi_mktcap   = black_litterman.market_implied_prior_returns(
                            market_caps=market_caps,
                            risk_aversion=2.5,
                            cov_matrix=S,
                            risk_free_rate=RISK_FREE_RATE,
                        )
                        raw_w_bl = _bl_weights(pi_mktcap, 1 / len(window))
                        w["Black-Litterman"] = dict(zip(live_tickers, raw_w_bl))

                    # ── Equal weight ──────────────────────────────────────────
                    if ENABLE_STRATEGIES["Equal_Weight"]:
                        w["Equal_Weight"] = dict(
                            zip(live_tickers, np.full(n_live, 1 / n_live)))

                    # ── LSTM predicted returns ─────────────────────────────────
                    lstm_mu = None
                    if ENABLE_STRATEGIES["LSTM_MVO"] or ENABLE_STRATEGIES["LSTM_BL"]:
                        raw_preds = predict_lstm_returns(live_tickers, date)
                        if len(raw_preds) >= 2:
                            # Align to live_tickers; fall back to EMA mu for missing
                            lstm_mu = mu.copy()
                            for t in live_tickers:
                                if t in raw_preds.index:
                                    lstm_mu[t] = raw_preds[t]

                    # ── LSTM MVO ──────────────────────────────────────────────
                    if ENABLE_STRATEGIES["LSTM_MVO"] and lstm_mu is not None:
                        w["LSTM_MVO"] = dict(zip(live_tickers,
                                                 regime_weights(lstm_mu, S, regime)))

                    # ── LSTM Black-Litterman ───────────────────────────────────
                    # Uses LSTM predictions as the prior (pi) in place of market-cap
                    # implied returns; sentiment views (P/Q) are the same as BL above.
                    if ENABLE_STRATEGIES["LSTM_BL"] and lstm_mu is not None:
                        try:
                            pi_lstm  = lstm_mu.reindex(live_tickers).fillna(
                                lstm_mu.mean() if len(lstm_mu) else 0.0)
                            raw_w_lb = _bl_weights(pi_lstm, 1 / len(window))
                            if raw_w_lb is not None:
                                w["LSTM_BL"] = dict(zip(live_tickers, raw_w_lb))
                        except Exception as e:
                            print(f"     ⚠ LSTM_BL failed: {e}")

                    # ── Transaction costs & record ─────────────────────────────
                    # Apply costs to today's equity value (carry yesterday forward
                    # then subtract turnover costs).
                    for s in STRATEGIES:
                        prev_val = equity.iloc[i - 1][s]
                        equity.loc[date, s] = apply_transaction_costs(
                            new_weights_dict=w[s],
                            old_weights_dict=prev_weights[s],
                            portfolio_value=prev_val,
                            all_tickers=all_valid_tickers,
                        )
                        prev_weights[s] = w[s].copy()
                        # Update numpy weight array for vectorised P&L
                        w_arr[s] = np.array([w[s].get(t, 0.0) for t in all_valid_tickers])

                    # ── Record weights (sparse) ────────────────────────────────
                    weight_dates.append(date)
                    for s in STRATEGIES:
                        weight_history[s].append({t: v for t, v in w[s].items() if v > 1e-6})

                    # ── Print summary ──────────────────────────────────────────
                    print(f"  ✓  {date.date()}  articles={len(articles)}"
                          f"  relative_views={P is not None}")
                    for s in STRATEGIES:
                        nz = {t: f"{v:.1%}" for t, v in w[s].items() if v > 0.001}
                        print(f"     {s:<22}: {nz}")

                except Exception as exc:
                    print(f"\n  ⚠  {date.date()} FAILED: {exc}")
                    traceback.print_exc()

            else:
                print(f"     Skipped: need {MIN_HISTORY_DAYS} rows and ≥2 tickers, "
                      f"got {len(window)} rows, {n_live} tickers")

        # ── DAILY P&L ─────────────────────────────────────────────────────────────
        # Run every day, including rebalance days.
        # On rebalance days equity.loc[date] was already set to the
        # cost-adjusted value above; we now apply today's market return ON TOP
        # of that cost-adjusted base (cost reduces principal, then market moves).
        day_returns       = returns_arr[i, :]
        day_returns_valid = np.where(np.isnan(day_returns[valid_idx]), 0.0, day_returns[valid_idx])
        for s in STRATEGIES:
            pnl  = float(np.dot(w_arr[s], day_returns_valid))
            base = equity.loc[date, s] if date in rebalance_dates else equity.iloc[i - 1][s]
            equity.loc[date, s] = base * (1 + pnl)

    print(f"\nBacktest complete.  Rebalances recorded: {len(weight_dates)}")

    # ─────────────────────────────────────────────
    # INDEX SAFETY FIX
    # ─────────────────────────────────────────────
    equity.index  = pd.DatetimeIndex(equity.index).tz_localize(None).normalize()
    equity        = equity.astype(float)

    os.makedirs("./metrics", exist_ok=True)

    # ─────────────────────────────────────────────
    # WEIGHT HISTORY HELPER (sparse dicts → DataFrame)
    # ─────────────────────────────────────────────
    def build_weight_df(strategy):
        if not weight_history[strategy]:
            return pd.DataFrame(columns=all_valid_tickers)
        idx = pd.DatetimeIndex(weight_dates).tz_localize(None).normalize()
        return pd.DataFrame(weight_history[strategy], index=idx).reindex(
            columns=all_valid_tickers, fill_value=0.0
        ).fillna(0.0)

    # ─────────────────────────────────────────────
    # WEIGHT STATISTICS LOG
    # ─────────────────────────────────────────────
    print("\n── Weight Statistics per Strategy ─────────────────────────────")

    weight_stats = {}

    for s in STRATEGIES:
        if not weight_history[s]:
            print(f"\n  {s}: no rebalance data")
            continue

        wdf = build_weight_df(s)
        weight_stats[s] = wdf

        stats = pd.DataFrame({
            "Mean  (%)": wdf.mean()       * 100,
            "Max   (%)": wdf.max()        * 100,
            "Min   (%)": wdf.min()        * 100,
            "StdDev(%)": wdf.std()        * 100,
            "# Non-Zero": (wdf > 0.001).sum(),
        })
        stats = stats[stats["# Non-Zero"] > 0].sort_values(
            "Mean  (%)", ascending=False)

        print(f"\n  ── {s} ──────────────────────────────────────────────────")
        print(stats.round(2).to_string())

    # ─────────────────────────────────────────────
    # REBALANCE-BY-REBALANCE WEIGHT LOG
    # ─────────────────────────────────────────────
    print("\n── Rebalance-by-Rebalance Weight Log ──────────────────────────")

    for s in STRATEGIES:
        if not weight_history[s]:
            continue
        print(f"\n  ── {s} ─────────────────────────────────────────────────")
        for date, w_dict in zip(weight_dates, weight_history[s]):
            nonzero = {t: f"{v:.1%}" for t, v in w_dict.items() if v > 0.001}
            print(f"    {date.date()}  →  {nonzero}")

    # ─────────────────────────────────────────────
    # SCALAR METRICS
    # ─────────────────────────────────────────────
    metrics_df = pd.DataFrame(
        {s: compute_metrics(equity[s]) for s in STRATEGIES}
    ).T

    print("\n── Performance Summary ────────────────────────────────────────")
    print(f"  Transaction costs : "
          f"{'ON  (%s bps)' % TRANSACTION_COST_BPS if ENABLE_TRANSACTION_COSTS else 'OFF'}")
    print(metrics_df.map(lambda x: f"{x:.4f}").to_string())

    # ─────────────────────────────────────────────
    # VISUALISATION SETUP
    # ─────────────────────────────────────────────
    COLORS       = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    STRAT_COLORS = dict(zip(STRATEGIES, COLORS))

    active_strategies = [s for s in STRATEGIES if s != "Equal_Weight"]
    TICKER_COLORS     = plt.cm.tab10(np.linspace(0, 0.9, len(all_valid_tickers)))

    def safe_plot(ax, series, **kwargs):
        ax.plot(series.index.to_numpy(), series.to_numpy(), **kwargs)

    # ─────────────────────────────────────────────
    # FIG 1 – Equity Curves
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for s in STRATEGIES:
        safe_plot(ax, equity[s], label=s, color=STRAT_COLORS[s], linewidth=2.0)
    tc_label = (f"TC: ON ({TRANSACTION_COST_BPS}bps)"
                if ENABLE_TRANSACTION_COSTS else "TC: OFF")
    ax.set_title(f"Portfolio Equity Curves  |  {tc_label}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value (base 100)")
    monthly_xaxis(ax)
    expand_xlim(ax)
    ax.legend()
    apply_layout(fig)
    plt.savefig("./metrics/fig1_equity_curves.png")

    # ─────────────────────────────────────────────
    # FIG 2 – Rolling Drawdown
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for s in STRATEGIES:
        dd = equity[s] / equity[s].cummax() - 1
        safe_plot(ax, dd * 100, label=s, color=STRAT_COLORS[s], linewidth=1.8)
    ax.set_title("Rolling Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    monthly_xaxis(ax)
    expand_xlim(ax)
    ax.legend()
    apply_layout(fig)
    plt.savefig("./metrics/fig2_drawdown.png")

    # ─────────────────────────────────────────────
    # FIG 3 – Rolling 21-day Sharpe
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for s in STRATEGIES:
        dr   = equity[s].pct_change().dropna()
        roll = (dr.rolling(21).mean() * 252 - RISK_FREE_RATE) / (
                dr.rolling(21).std() * np.sqrt(252))
        safe_plot(ax, roll, label=s, color=STRAT_COLORS[s], linewidth=1.8)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Rolling 21-day Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    monthly_xaxis(ax)
    expand_xlim(ax)
    ax.legend()
    apply_layout(fig)
    plt.savefig("./metrics/fig3_rolling_sharpe.png")

    # ─────────────────────────────────────────────
    # FIG 4 – Rolling 21-day CVaR
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for s in STRATEGIES:
        dr        = equity[s].pct_change().dropna()
        cvar_roll = dr.rolling(21).apply(
            lambda x: x[x <= np.quantile(x, CVAR_ALPHA)].mean(), raw=True)
        safe_plot(ax, cvar_roll * 100, label=s, color=STRAT_COLORS[s], linewidth=1.8)
    ax.set_title(f"Rolling 21-day CVaR (α={CVAR_ALPHA:.0%})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("CVaR (%)")
    monthly_xaxis(ax)
    expand_xlim(ax)
    ax.legend()
    apply_layout(fig)
    plt.savefig("./metrics/fig4_rolling_cvar.png")

    # ─────────────────────────────────────────────
    # FIG 5 – Rolling 21-day Volatility
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for s in STRATEGIES:
        dr   = equity[s].pct_change().dropna()
        rvol = dr.rolling(21).std() * np.sqrt(252) * 100
        safe_plot(ax, rvol, label=s, color=STRAT_COLORS[s], linewidth=1.8)
    ax.set_title("Rolling 21-day Annualised Volatility",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Volatility (%)")
    monthly_xaxis(ax)
    expand_xlim(ax)
    ax.legend()
    apply_layout(fig)
    plt.savefig("./metrics/fig5_rolling_vol.png")

    # ─────────────────────────────────────────────
    # FIG 6 – Portfolio Weights per Rebalance
    # ─────────────────────────────────────────────
    if active_strategies:
        n_strats = len(active_strategies)
        fig, axes = plt.subplots(
            n_strats, 1,
            figsize=(FIG_WEIGHT[0], FIG_WEIGHT[1] * n_strats),
            sharex=False,
        )
        if n_strats == 1:
            axes = [axes]

        for ax, s in zip(axes, active_strategies):
            wdf = build_weight_df(s)
            if wdf.empty:
                ax.text(0.5, 0.5, f"{s}\nNo rebalance data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(f"{s} – Weights at Each Rebalance",
                             fontsize=12, fontweight="bold")
                continue

            x_pos  = np.arange(len(wdf))
            bottom = np.zeros(len(wdf))
            for j, ticker in enumerate(all_valid_tickers):
                col = (wdf[ticker].values
                       if ticker in wdf.columns else np.zeros(len(wdf)))
                ax.bar(x_pos, col, bottom=bottom,
                       label=ticker, color=TICKER_COLORS[j],
                       width=0.75, edgecolor="white", linewidth=0.5)
                bottom += col

            # Annotate segments > 4%
            bottom_ann = np.zeros(len(wdf))
            for j, ticker in enumerate(all_valid_tickers):
                col = (wdf[ticker].values
                       if ticker in wdf.columns else np.zeros(len(wdf)))
                for xi, (val, bot) in enumerate(zip(col, bottom_ann)):
                    if val > 0.04:
                        ax.text(xi, bot + val / 2, f"{val:.0%}",
                                ha="center", va="center",
                                fontsize=7, color="white", fontweight="bold")
                bottom_ann += col

            ax.set_title(f"{s} – Portfolio Weights at Each Rebalance",
                         fontsize=12, fontweight="bold")
            ax.set_ylabel("Weight")
            ax.set_ylim(0, 1.08)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
            rebalance_xaxis(ax, wdf.index, rotation=45, fontsize=7)
            ax.legend(loc="upper left", fontsize=7,
                      ncol=min(len(all_valid_tickers), 10),
                      framealpha=0.7)
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle("Portfolio Weights at Each Rebalance Date",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.subplots_adjust(bottom=0.15, hspace=0.45)
        plt.savefig("./metrics/fig6_weights.png")
    
    # ─────────────────────────────────────────────
    # FIG 7 – Summary Bar Chart
    # ─────────────────────────────────────────────
    scalar_cols = ["Annual Return", "Annual Volatility", "Sharpe Ratio",
                   "Max Drawdown", "Calmar Ratio", "CVaR (95%)"]
    fig, axes   = plt.subplots(2, 3, figsize=FIG_SQUARE)

    for idx, col in enumerate(scalar_cols):
        ax     = axes.flatten()[idx]
        values = metrics_df[col].astype(float)
        bars   = ax.bar(STRATEGIES, values,
                        color=[STRAT_COLORS[s] for s in STRATEGIES],
                        edgecolor="white", linewidth=0.6)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels(STRATEGIES, rotation=25, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.001 if val >= 0 else -0.005),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Summary Performance Metrics  |  "
        f"TC: {'ON (%sbps)' % TRANSACTION_COST_BPS if ENABLE_TRANSACTION_COSTS else 'OFF'}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(pad=2.5)
    plt.savefig("./metrics/fig7_summary_metrics.png")

    # ─────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────
    equity.to_csv("./metrics/equity_curves.csv")
    metrics_df.to_csv("./metrics/performance_metrics.csv")
    pd.DataFrame(news_log).to_csv("./metrics/news_log.csv", index=False)

    # ── Rebalance weights log ──────────────────────────────────────────────────
    rebalance_rows = []
    for rebal_date, w_dicts in zip(weight_dates, zip(*[weight_history[s] for s in STRATEGIES])):
        for s, w_dict in zip(STRATEGIES, w_dicts):
            for ticker, weight in w_dict.items():
                rebalance_rows.append({
                    "date":     rebal_date.date(),
                    "strategy": s,
                    "ticker":   ticker,
                    "weight":   round(weight, 6),
                })
    pd.DataFrame(rebalance_rows).to_csv("./metrics/rebalance_weights.csv", index=False)

    print("\n✓ CSVs: ./metrics/equity_curves.csv | ./metrics/performance_metrics.csv | ./metrics/news_log.csv | ./metrics/rebalance_weights.csv")
    print("✓ Figs: fig1–fig7")


if __name__ == "__main__":
    main()