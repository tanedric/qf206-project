import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from scipy.optimize import minimize
from pypfopt import black_litterman, risk_models, expected_returns, EfficientFrontier
from pypfopt import objective_functions
import matplotlib.pyplot as plt
import warnings
import logging, contextlib, io
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os
from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# WARNING FILTERS
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
API_KEY            = os.getenv("STOCKNEWS_API_KEY") 
TICKERS            = [
    'CWEN', 'NEE', 'BEP', 'FSLR', 'BE',
    'SCHN', 'TT', 'PLUG', 'FLNC', 'SEDG', 'VST', 'AES', 'ED', 'SMR', 'ARRY', "^GSPC", "ICLN"
]

START_DATE         = "2016-01-01"
END_DATE           = "2026-01-01"
REBALANCE_FREQ     = "ME"
NEWS_LOOKBACK_DAYS = 15
SENTIMENT_SCALING  = 0.3
RISK_FREE_RATE     = 0.02
CVAR_ALPHA         = 0.05
MIN_HISTORY_DAYS   = 30

# ── STRATEGY TOGGLES ──────────────────────────
ENABLE_MVO              = True
ENABLE_BLACK_LITTERMAN  = False
ENABLE_EQUAL_WEIGHT     = False
ENABLE_ENTROPY_POOLING  = False

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
    "Entropy_Pooling":   ENABLE_ENTROPY_POOLING,
}
STRATEGIES = [s for s, enabled in ENABLE_STRATEGIES.items() if enabled]

TRANSACTION_COST = TRANSACTION_COST_BPS / 10_000

# ─────────────────────────────────────────────
# SILENT DOWNLOAD HELPER
# ─────────────────────────────────────────────
def silent_download(tickers, start, end):
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

print("── Ticker Classification ───────────────────────────────────────")
active_tickers, ipo_map, delisted_tickers = classify_tickers(
    TICKERS, START_DATE, END_DATE)
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
def fetch_news(tickers, start, end, max_pages=1):
    """Only called when Black-Litterman or Entropy Pooling is active."""
    articles = []
    for page in range(1, max_pages + 1):
        try:
            r = requests.get(
                "https://stocknewsapi.com/api/v1",
                params={
                    "tickers": ",".join(tickers), "items": 50,
                    "page": page,
                    "date": f"{start.strftime('%m%d%Y')}-{end.strftime('%m%d%Y')}",
                    "token": API_KEY,
                },
                timeout=10,
            )
            batch = r.json().get("data", [])
            articles.extend(batch)
            if not batch:
                break
        except Exception:
            break
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
# ENTROPY POOLING
# ─────────────────────────────────────────────
def entropy_pooling_returns(returns_window, views, tickers):
    view_assets = [t for t, v in views.items() if v != 0]
    if not view_assets:
        return None
    n   = len(returns_window)
    p0  = np.ones(n) / n
    A   = returns_window[view_assets].values
    b   = np.array([views[t] for t in view_assets])

    def dual(v):
        x   = -v @ A.T
        M   = x.max()
        lse = M + np.log(np.sum(p0 * np.exp(x - M)))
        return lse + v @ b

    v_opt  = minimize(dual, np.zeros(len(view_assets)), method="SLSQP").x
    x      = -v_opt @ A.T
    M      = x.max()
    log_p  = np.log(p0) + x - (M + np.log(np.sum(p0 * np.exp(x - M))))
    p_post = np.exp(log_p)
    return pd.Series(p_post @ returns_window.values, index=tickers)

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
def monthly_xaxis(ax, rotation=45, fontsize=8):
    """Monthly major ticks, weekly minor ticks for grid reference."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    ax.grid(which="minor", axis="x", alpha=0.12, linestyle=":")
    ax.grid(which="major", axis="x", alpha=0.25)
    plt.setp(ax.get_xticklabels(),
             rotation=rotation, ha="center", fontsize=fontsize)


def rebalance_xaxis(ax, dates, rotation=45, fontsize=8):
    """One tick per rebalance date, two-line label."""
    x_pos  = np.arange(len(dates))
    labels = [d.strftime("%b\n%Y") for d in dates]
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

equity         = pd.DataFrame(100.0, index=returns.index, columns=STRATEGIES)
news_log       = []

print(f"\nStarting backtest …")
print(f"  Strategies        : {STRATEGIES}")
print(f"  Transaction costs : {'ON  (%s bps one-way)' % TRANSACTION_COST_BPS if ENABLE_TRANSACTION_COSTS else 'OFF'}\n")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
for i in range(1, len(returns)):
    date    = returns.index[i]
    av_tick = available_tickers(date, all_valid_tickers, ipo_map)

    # ── REBALANCE BLOCK ───────────────────────────────────────────────────────
    # Indented correctly inside the for-loop.
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

                S  = risk_models.CovarianceShrinkage(
                         window, returns_data=True).ledoit_wolf()
                mu = expected_returns.ema_historical_return(
                         window, returns_data=True, span=180)

                # ── MVO ───────────────────────────────────────────────────
                if ENABLE_STRATEGIES["MVO"]:
                    w["MVO"] = dict(zip(live_tickers,
                                        max_sharpe_weights(mu, S)))

                # ── News & sentiment (only when needed) ───────────────────
                needs_news = (ENABLE_STRATEGIES["Black-Litterman"] or
                              ENABLE_STRATEGIES["Entropy_Pooling"])
                if needs_news:
                    articles = fetch_news(
                        live_tickers,
                        date - pd.Timedelta(days=NEWS_LOOKBACK_DAYS),
                        date,
                    )
                    news_log.extend({
                        "rebalance_date": date.date(),
                        "title":     a.get("title"),
                        "tickers":   a.get("tickers"),
                        "sentiment": a.get("sentiment"),
                        "date":      a.get("date"),
                    } for a in articles)

                    views, omega_vec = sentiment_views(articles, live_tickers)
                    P, Q             = build_relative_views(views, live_tickers)

                # ── Black-Litterman ───────────────────────────────────────
                if ENABLE_STRATEGIES["Black-Litterman"]:
                    market_caps = MARKET_CAPS_FULL.reindex(live_tickers).fillna(1)
                    pi  = black_litterman.market_implied_prior_returns(
                        market_caps=market_caps,
                        risk_aversion=2.5,
                        cov_matrix=S,
                        risk_free_rate=RISK_FREE_RATE,
                    )
                    tau     = 1 / len(window)
                    idx_map = {t: j for j, t in enumerate(live_tickers)}

                    print(f"     Views        : { {k: round(v,4) for k,v in views.items()} }")
                    print(f"     Relative P/Q : {P is not None} "
                          f"({'%d pairs' % len(Q) if Q is not None else 'none'})")

                    if P is not None and Q is not None:
                        view_omegas = []
                        for row in P:
                            involved = [j for j, v in enumerate(row) if v != 0]
                            view_omegas.append(
                                np.mean([omega_vec[j] for j in involved]))
                        omega_matrix = np.diag(view_omegas).astype(float)
                        bl_model     = black_litterman.BlackLittermanModel(
                            S, pi=pi, P=P, Q=Q,
                            omega=omega_matrix, tau=tau,
                        )
                    else:
                        active_views = {t: v for t, v in views.items() if v != 0}
                        if active_views:
                            active_omega = np.diag([
                                omega_vec[idx_map[t]] for t in active_views
                            ]).astype(float)
                            bl_model = black_litterman.BlackLittermanModel(
                                S, pi=pi,
                                absolute_views=active_views,
                                omega=active_omega,
                                tau=tau,
                            )
                        else:
                            print("     No views — using prior only")
                            bl_model = black_litterman.BlackLittermanModel(
                                S, pi=pi, tau=tau)

                    ret_bl = bl_model.bl_returns()
                    if np.all(np.isfinite(ret_bl)):
                        try:
                            ef_bl = EfficientFrontier(ret_bl, S, weight_bounds=(0, 1))
                            ef_bl.add_objective(objective_functions.L2_reg, gamma=0.1)
                            ef_bl.max_sharpe(risk_free_rate=RISK_FREE_RATE)
                            raw_w_bl = np.array(list(ef_bl.clean_weights().values()))
                        except Exception:
                            ef_bl    = EfficientFrontier(ret_bl, S, weight_bounds=(0, 1))
                            ef_bl.min_volatility()
                            raw_w_bl = np.array(list(ef_bl.clean_weights().values()))
                        w["Black-Litterman"] = dict(zip(live_tickers, raw_w_bl))
                    else:
                        print("     ⚠ BL returns not finite — keeping previous weights")

                # ── Entropy Pooling ───────────────────────────────────────
                if ENABLE_STRATEGIES["Entropy_Pooling"]:
                    ep_mu = entropy_pooling_returns(window, views, live_tickers)
                    if ep_mu is not None:
                        w["Entropy_Pooling"] = dict(
                            zip(live_tickers, max_sharpe_weights(ep_mu, S)))

                # ── Equal weight ──────────────────────────────────────────
                if ENABLE_STRATEGIES["Equal_Weight"]:
                    w["Equal_Weight"] = dict(
                        zip(live_tickers, np.full(n_live, 1 / n_live)))

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

                # ── Record weights ─────────────────────────────────────────
                weight_dates.append(date)
                for s in STRATEGIES:
                    full_w = np.array([w[s].get(t, 0.0)
                                       for t in all_valid_tickers])
                    weight_history[s].append(full_w)

                # ── Print summary ──────────────────────────────────────────
                print(f"  ✓  {date.date()}  articles={len(articles)}"
                      f"  relative_views={P is not None}")
                for s in STRATEGIES:
                    nz = {t: f"{v:.1%}"
                          for t, v in zip(all_valid_tickers,
                                          weight_history[s][-1])
                          if v > 0.001}
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
    for s in STRATEGIES:
        pnl = sum(
            w[s].get(t, 0.0) * (
                returns.loc[date, t]
                if not np.isnan(returns.loc[date, t]) else 0.0
            )
            for t in all_valid_tickers
        )
        # On rebalance days equity.loc[date, s] already holds the
        # cost-adjusted starting value; grow it by today's return.
        # On non-rebalance days use yesterday's closing value.
        base = equity.loc[date, s] if date in rebalance_dates else equity.iloc[i - 1][s]
        equity.loc[date, s] = base * (1 + pnl)

print(f"\nBacktest complete.  Rebalances recorded: {len(weight_dates)}")

# ─────────────────────────────────────────────
# INDEX SAFETY FIX
# ─────────────────────────────────────────────
equity.index  = pd.DatetimeIndex(equity.index).tz_localize(None).normalize()
equity        = equity.astype(float)

# ─────────────────────────────────────────────
# WEIGHT STATISTICS LOG
# ─────────────────────────────────────────────
print("\n── Weight Statistics per Strategy ─────────────────────────────")

weight_stats = {}

for s in STRATEGIES:
    if not weight_history[s]:
        print(f"\n  {s}: no rebalance data")
        continue

    wdf = pd.DataFrame(
        weight_history[s],
        index=pd.DatetimeIndex(weight_dates).tz_localize(None).normalize(),
        columns=all_valid_tickers,
    )

    stats = pd.DataFrame({
        "Mean  (%)": wdf.mean()       * 100,
        "Max   (%)": wdf.max()        * 100,
        "Min   (%)": wdf.min()        * 100,
        "StdDev(%)": wdf.std()        * 100,
        "# Non-Zero": (wdf > 0.001).sum(),
    })
    stats       = stats[stats["# Non-Zero"] > 0].sort_values(
        "Mean  (%)", ascending=False)
    weight_stats[s] = wdf

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
    for date, w_arr in zip(weight_dates, weight_history[s]):
        nonzero = {t: f"{v:.1%}"
                   for t, v in zip(all_valid_tickers, w_arr)
                   if v > 0.001}
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
# WEIGHT HISTORY HELPER
# ─────────────────────────────────────────────
def build_weight_df(strategy):
    if not weight_history[strategy]:
        return pd.DataFrame(columns=all_valid_tickers)
    return pd.DataFrame(
        weight_history[strategy],
        index=pd.DatetimeIndex(weight_dates).tz_localize(None).normalize(),
        columns=all_valid_tickers,
    )

# ─────────────────────────────────────────────
# VISUALISATION SETUP
# ─────────────────────────────────────────────
COLORS       = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
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
plt.savefig("fig1_equity_curves.png")
plt.show()

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
plt.savefig("fig2_drawdown.png")
plt.show()

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
plt.savefig("fig3_rolling_sharpe.png")
plt.show()

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
plt.savefig("fig4_rolling_cvar.png")
plt.show()

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
plt.savefig("fig5_rolling_vol.png")
plt.show()

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
    plt.savefig("fig6_weights.png")
    plt.show()

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
plt.savefig("fig7_summary_metrics.png")
plt.show()

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
equity.to_csv("equity_curves.csv")
metrics_df.to_csv("performance_metrics.csv")
pd.DataFrame(news_log).to_csv("news_log.csv", index=False)
print("\n✓ CSVs: equity_curves.csv | performance_metrics.csv | news_log.csv")
print("✓ Figs: fig1–fig7")