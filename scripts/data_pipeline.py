"""
data_pipeline.py
----------------
Standalone data pipeline for the 12-indicator LSTM stock return model.
Fetches raw data via yfinance and pandas_datareader, computes all features,
normalises with pre-computed mu/sigma, and produces (x, y) sequences.

Usage
-----
    from data_pipeline import compute_indicators, get_seqs, z_score_norm
    import json

    with open("model/param_dict.json") as f:
        raw = json.load(f)
    param_dict = {k: tuple(v) for k, v in raw.items()}

    df  = compute_indicators(ticker="AAPL", start_date="2017-01-01", end_date="2020-01-01")
    df  = df.apply(z_score_norm, axis=0)   # uses global param_dict
    x, y = get_seqs(df, seq_len=9)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime
import warnings

warnings.filterwarnings("ignore")

# Module-level cache so each ticker's sharesOutstanding is fetched at most once
# per process, avoiding repeated yfinance info calls that trigger rate limits.
_shares_cache: dict = {}

# ── Feature functions ─────────────────────────────────────────────────────────
def fetch_price_data(ticker: str, raw_start: str, end_date: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a ticker from yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    raw_start : str
        Start date string 'YYYY-MM-DD' (includes lookback buffer).
    end_date : str
        End date string 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV DataFrame with a DatetimeIndex.
    """
    raw = yf.download(ticker, start=raw_start, end=end_date, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for {ticker} between {raw_start} and {end_date}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.to_datetime(raw.index)
    return raw.sort_index()


def fetch_shares_outstanding(ticker: str) -> float or None:
    """
    Retrieve shares outstanding from yfinance ticker info.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    float or None
        Shares outstanding, or None if unavailable.
    """
    if ticker in _shares_cache:
        return _shares_cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        value = info.get("sharesOutstanding", None)
    except Exception as e:
        print(f"    Warning: Could not fetch sharesOutstanding: {e}")
        value = None
    _shares_cache[ticker] = value
    return value


def fetch_ff5_factors(raw_start: str, end_date: str) -> pd.DataFrame or None:
    """
    Download the Fama-French 5-Factor monthly data from Kenneth French's library.

    Parameters
    ----------
    raw_start : str
        Start date string 'YYYY-MM-DD'.
    end_date : str
        End date string 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame or None
        Monthly FF5 factor DataFrame (decimal), or None if download fails.
    """
    try:
        import io, zipfile, requests as _requests
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
        resp = _requests.get(url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv"))
            raw_text = zf.read(csv_name).decode("latin-1")
        # The file has a header block before the data and a footer after — find the data rows
        lines = raw_text.splitlines()
        # Find header line (contains Mkt-RF)
        header_idx = next(i for i, l in enumerate(lines) if "Mkt-RF" in l)
        # Data ends at the first blank line after the header
        data_lines = []
        for l in lines[header_idx + 1:]:
            if l.strip() == "" or l.strip().startswith("Annual"):
                break
            data_lines.append(l)
        csv_block = lines[header_idx] + "\n" + "\n".join(data_lines)
        ff5 = pd.read_csv(io.StringIO(csv_block), index_col=0)
        ff5.index = pd.to_datetime(ff5.index.astype(str).str.strip(), format="%Y%m")
        ff5.index = ff5.index + pd.offsets.MonthEnd(0)
        ff5 = ff5.apply(pd.to_numeric, errors="coerce") / 100.0
        ff5.index = ff5.index.normalize()
        ff5 = ff5.loc[raw_start:end_date]
        return ff5
    except Exception as e:
        print(f"    Warning: Could not download FF5 data: {e}")
        return None


def fetch_industry_tracking_stock(
    raw_start: str, end_date: str, etf: str = "QQQ"
) -> pd.Series:
    """
    Download monthly returns for an industry tracking ETF.

    Parameters
    ----------
    raw_start : str
        Start date string 'YYYY-MM-DD'.
    end_date : str
        End date string 'YYYY-MM-DD'.
    etf : str
        ETF ticker symbol (default: 'QQQ' for tech/Nasdaq).

    Returns
    -------
    pd.Series
        Monthly return series for the ETF.
    """
    raw = yf.download(etf, start=raw_start, end=end_date, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    monthly_close = raw["Close"].resample("ME").last()
    return monthly_close.pct_change().rename(etf)


def fetch_market_returns(raw_start: str, end_date: str, etf: str = "SPY") -> pd.Series:
    """
    Download monthly returns for a market proxy ETF (used in CAPM fallback).

    Parameters
    ----------
    raw_start : str
        Start date string 'YYYY-MM-DD'.
    end_date : str
        End date string 'YYYY-MM-DD'.
    etf : str
        Market proxy ETF (default: 'SPY').

    Returns
    -------
    pd.Series
        Monthly return series for the market proxy.
    """
    raw = yf.download(etf, start=raw_start, end=end_date, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    monthly_close = raw["Close"].resample("ME").last()
    return monthly_close.pct_change().rename("market")


# =============================================================================
# Monthly Base Series Helpers
# =============================================================================


def compute_monthly_close(daily_data: pd.DataFrame) -> pd.Series:
    """Resample daily close prices to month-end."""
    return daily_data["Close"].resample("ME").last()


def compute_monthly_returns(monthly_close: pd.Series) -> pd.Series:
    """Compute simple monthly returns from month-end close prices."""
    return np.log(1+monthly_close.pct_change()).rename("returns")


# =============================================================================
# Individual Indicator Functions
# =============================================================================


def compute_mom1m(monthly_returns: pd.Series) -> pd.Series:
    """
    Compute 1-month momentum (prior month's return).

    Definition
    ----------
    mom1m_t = return_{t-1}

    Skips the most recent month to avoid microstructure reversal effects,
    consistent with standard cross-sectional momentum literature
    (Jegadeesh & Titman, 1993).

    Parameters
    ----------
    monthly_returns : pd.Series
        Series of monthly simple returns.

    Returns
    -------
    pd.Series
        1-month momentum series.
    """
    return monthly_returns.shift(1).rename("mom1m")


def compute_mvel(
    monthly_close: pd.Series, shares_outstanding: float or None
) -> pd.Series:
    """
    Compute log market capitalisation (size).

    Definition
    ----------
    mvel_t = log(Close_t * SharesOutstanding)

    Parameters
    ----------
    monthly_close : pd.Series
        Month-end close prices.
    shares_outstanding : float or None
        Total shares outstanding (static snapshot from yfinance info).

    Returns
    -------
    pd.Series
        Log market capitalisation series, or NaN series if shares unavailable.
    """
    if shares_outstanding is None:
        print("    Warning: sharesOutstanding not available; mvel will be NaN.")
        return pd.Series(np.nan, index=monthly_close.index, name="mvel")

    mkt_cap = monthly_close * shares_outstanding
    return np.log(mkt_cap.replace(0, np.nan)).rename("mvel")


def compute_zerotrade(daily_data: pd.DataFrame) -> pd.Series:
    """
    Compute zero-trading-days ratio.

    Definition
    ----------
    zerotrade_t = (# days with zero volume in month t) / (total trading days in month t)

    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily OHLCV DataFrame.

    Returns
    -------
    pd.Series
        Monthly fraction of zero-volume days.
    """
    volume = daily_data["Volume"]
    zero_days = (volume == 0).resample("ME").sum()
    total_days = volume.resample("ME").count()
    return (zero_days / total_days).rename("zerotrade")


def compute_dolvol(daily_data: pd.DataFrame) -> pd.Series:
    """
    Compute log average daily dollar volume.

    Definition
    ----------
    dolvol_t = log( mean_d( Close_d * Volume_d ) ) for days d in month t

    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily OHLCV DataFrame.

    Returns
    -------
    pd.Series
        Log average daily dollar volume series.
    """
    dollar_vol_daily = daily_data["Close"] * daily_data["Volume"]
    monthly_mean = dollar_vol_daily.resample("ME").mean()
    return np.log(monthly_mean.replace(0, np.nan)).rename("dolvol")


def compute_ill(daily_data: pd.DataFrame) -> pd.Series:
    """
    Compute Amihud (2002) illiquidity ratio.

    Definition
    ----------
    ILLIQ_t = mean_d( |R_d| / DVOL_d )  for days d in month t

    where R_d is the daily return and DVOL_d = Close_d * Volume_d.

    Reference: Amihud, Y. (2002). Illiquidity and stock returns:
    Cross-section and time-series effects. Journal of Financial Markets, 5, 31-56.

    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily OHLCV DataFrame.

    Returns
    -------
    pd.Series
        Monthly Amihud illiquidity series.
    """
    daily_ret = daily_data["Close"].pct_change()
    dollar_vol = daily_data["Close"] * daily_data["Volume"]
    ratio = daily_ret.abs() / dollar_vol.replace(0, np.nan)
    return ratio.resample("ME").mean().rename("ill")


def compute_retvol(daily_data: pd.DataFrame) -> pd.Series:
    """
    Compute return volatility (standard deviation of daily returns within each month).

    Definition
    ----------
    retvol_t = std_d( R_d )  for days d in month t

    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily OHLCV DataFrame.

    Returns
    -------
    pd.Series
        Monthly return volatility series.
    """
    daily_ret = daily_data["Close"].pct_change()
    return daily_ret.resample("ME").std().rename("retvol")


def compute_mom6m(monthly_returns: pd.Series) -> pd.Series:
    """
    Compute 6-month momentum.

    Definition
    ----------
    mom6m_t = cumulative return over months [t-7, t-2]
            = prod(1 + R_{t-7} ... R_{t-2}) - 1

    Skips month t-1 (most recent) to avoid short-term reversal,
    consistent with Jegadeesh & Titman (1993).

    Parameters
    ----------
    monthly_returns : pd.Series
        Series of monthly simple returns.

    Returns
    -------
    pd.Series
        6-month momentum series.
    """
    return monthly_returns.rolling(window=6).sum().shift(1).rename("mom6m")


def compute_chmom(mom6m: pd.Series) -> pd.Series:
    """
    Compute change in 6-month momentum.

    Definition
    ----------
    chmom_t = mom6m_t - mom6m_{t-6}

    Captures acceleration or deceleration in intermediate-term momentum.
    Reference: Gettleman & Marks (2006).

    Parameters
    ----------
    mom6m : pd.Series
        6-month momentum series (output of compute_mom6m).

    Returns
    -------
    pd.Series
        Change-in-momentum series.
    """
    return (mom6m.pct_change()).rename("chmom")


def compute_mom12m(monthly_returns: pd.Series) -> pd.Series:
    """
    Compute 12-month momentum.

    Definition
    ----------
    mom12m_t = cumulative return over months [t-13, t-2]
             = prod(1 + R_{t-13} ... R_{t-2}) - 1

    Skips month t-1 to avoid short-term reversal.

    Parameters
    ----------
    monthly_returns : pd.Series
        Series of monthly simple returns.

    Returns
    -------
    pd.Series
        12-month momentum series.
    """
    return monthly_returns.rolling(window=12).sum().shift(1).rename("mom12m")


def compute_mom36m(monthly_returns: pd.Series) -> pd.Series:
    """
    Compute 36-month momentum (long-term reversal signal).

    Definition
    ----------
    mom36m_t = cumulative return over months [t-37, t-13]
             = prod(1 + R_{t-37} ... R_{t-13}) - 1

    Skips the most recent 12 months to isolate long-term effects,
    consistent with De Bondt & Thaler (1985).

    Parameters
    ----------
    monthly_returns : pd.Series
        Series of monthly simple returns.

    Returns
    -------
    pd.Series
        36-month (long-term) momentum series.
    """
    return monthly_returns.rolling(window=36).sum().shift(1).rename("mom36m")


def compute_indmom(industry_returns: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    # Normalize both indexes to month-end so reindex always finds a match
    industry_returns.index = industry_returns.index.to_period("M").to_timestamp("M")
    target_index_norm = target_index.to_period("M").to_timestamp("M")

    indmom = industry_returns.reindex(target_index_norm).shift(1)
    indmom.index = target_index  # restore original index
    return indmom.rename("indmom")


def compute_idiovol_ff5(
    monthly_returns: pd.Series,
    ff5: pd.DataFrame,
    window: int = 36,
) -> pd.Series:
    """
    Compute idiosyncratic volatility using a rolling Fama-French 5-Factor regression.

    Definition
    ----------
    For each month t, regress excess stock returns on FF5 factors using the
    prior `window` months, then compute the standard deviation of residuals:

        idiovol_t = std( epsilon_t )
        where epsilon = R_excess - (alpha + b1*MktRF + b2*SMB + b3*HML
                                           + b4*RMW + b5*CMA)

    Reference: Ang et al. (2006). The Cross-Section of Volatility and
    Expected Returns. Journal of Finance, 61(1), 259-299.

    Parameters
    ----------
    monthly_returns : pd.Series
        Monthly simple returns of the stock.
    ff5 : pd.DataFrame
        Fama-French 5-Factor monthly data (must include 'RF' and factor columns).
    window : int
        Rolling window in months (default: 36).

    Returns
    -------
    pd.Series
        Monthly idiosyncratic volatility series.
    """
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    stock_excess = monthly_returns - ff5["RF"].reindex(monthly_returns.index, method="nearest")
    ff5_aligned = ff5[factor_cols].reindex(monthly_returns.index, method="nearest")
    reg_data = pd.concat([stock_excess.rename("excess"), ff5_aligned], axis=1).dropna()

    idiovol_dict = {}

    for i in range(window - 1, len(reg_data)):
        window_data = reg_data.iloc[i - window + 1 : i + 1]
        y = window_data["excess"].values
        X = np.column_stack([np.ones(len(window_data)), window_data[factor_cols].values])

        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ coef
            std_resid = np.std(residuals, ddof=X.shape[1])
        except Exception:
            std_resid = np.nan

        idiovol_dict[reg_data.index[i]] = std_resid

    return pd.Series(idiovol_dict, name="idiovol")


def compute_idiovol_capm(
    monthly_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 36,
) -> pd.Series:
    """
    Compute idiosyncratic volatility using a rolling CAPM regression (fallback).

    Definition
    ----------
    For each month t, regress stock returns on market returns using prior
    `window` months, then compute the standard deviation of residuals:

        idiovol_t = std( epsilon_t )
        where epsilon = R_stock - (alpha + beta * R_market)

    Parameters
    ----------
    monthly_returns : pd.Series
        Monthly simple returns of the stock.
    market_returns : pd.Series
        Monthly simple returns of the market proxy (e.g., SPY).
    window : int
        Rolling window in months (default: 36).

    Returns
    -------
    pd.Series
        Monthly idiosyncratic volatility series.
    """
    print("  [compute_idiovol_capm] Computing idiovol via CAPM rolling regression ...")
    reg_data = pd.concat(
        [monthly_returns.rename("stock"), market_returns],
        axis=1,
    ).dropna()

    idiovol_dict = {}

    for i in range(window - 1, len(reg_data)):
        window_data = reg_data.iloc[i - window + 1 : i + 1]
        y = window_data["stock"].values
        X = np.column_stack([np.ones(len(window_data)), window_data["market"].values])

        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ coef
            std_resid = np.std(residuals, ddof=2)
        except Exception:
            std_resid = np.nan

        idiovol_dict[reg_data.index[i]] = std_resid

    return pd.Series(idiovol_dict, name="idiovol")


# =============================================================================
# Orchestrator
# =============================================================================


def compute_indicators(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    industry_etf: str = "QQQ",
    market_etf: str = "SPY",
    idiovol_window: int = 36,
    daily_data: pd.DataFrame = None,
    ff5: pd.DataFrame = None,
    industry_returns: pd.Series = None,
    market_returns: pd.Series = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date : str, optional
        Start date 'YYYY-MM-DD'. Defaults to the stock's IPO date.
    end_date : str, optional
        End date 'YYYY-MM-DD'. Defaults to today.
    industry_etf : str
        ETF used as industry benchmark for indmom (default: 'QQQ').
    market_etf : str
        ETF used as market proxy for CAPM fallback (default: 'SPY').
    idiovol_window : int
        Rolling window (months) for idiovol regression (default: 36).
    daily_data : pd.DataFrame, optional
        Pre-fetched daily OHLCV data. If provided, skips the price download.
    ff5 : pd.DataFrame, optional
        Pre-fetched FF5 factor data. If provided, skips the FF5 download.
    industry_returns : pd.Series, optional
        Pre-fetched industry ETF returns. If provided, skips the ETF download.
    market_returns : pd.Series, optional
        Pre-fetched market returns. Used as CAPM fallback if ff5 is None.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame indexed by date with all indicators as columns.
    """

    # Resolve dates
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    if start_date is None:
        raise ValueError(f"start_date is required for compute_indicators({ticker})")

    # Extra historical buffer needed for the longest lookback (36m + buffer)
    lookback_years = 4
    raw_start = (
        pd.Timestamp(start_date) - pd.DateOffset(years=lookback_years)
    ).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Computing indicators for: {ticker}")
    print(f"  Date range : {start_date} -> {end_date}")
    print(f"  Raw data from: {raw_start}")
    print(f"{'='*60}\n")

    # Fetch any data not supplied by the caller
    if daily_data is None:
        daily_data = fetch_price_data(ticker, raw_start, end_date)
    shares_out = fetch_shares_outstanding(ticker)
    if ff5 is None:
        ff5 = fetch_ff5_factors(raw_start, end_date)
    if industry_returns is None:
        industry_returns = fetch_industry_tracking_stock(raw_start, end_date, etf=industry_etf)
    if market_returns is None and ff5 is None:
        market_returns = fetch_market_returns(raw_start, end_date, etf=market_etf)

    # Build monthly base series
    monthly_close = compute_monthly_close(daily_data)
    monthly_returns = compute_monthly_returns(monthly_close)

    # Compute each indicator

    mom1m   = compute_mom1m(monthly_returns)
    mvel    = compute_mvel(monthly_close, shares_out)
    zerotrade = compute_zerotrade(daily_data)
    dolvol  = compute_dolvol(daily_data)
    ill     = compute_ill(daily_data)
    retvol  = compute_retvol(daily_data)
    mom6m   = compute_mom6m(monthly_returns)
    chmom   = compute_chmom(mom6m)
    mom12m  = compute_mom12m(monthly_returns)
    mom36m  = compute_mom36m(monthly_returns)
    indmom  = compute_indmom(industry_returns, monthly_close.index)

    if ff5 is not None:
        idiovol = compute_idiovol_ff5(monthly_returns, ff5, window=idiovol_window)
    else:
        print("  FF5 unavailable; falling back to CAPM for idiovol.")
        idiovol = compute_idiovol_capm(monthly_returns, market_returns, window=idiovol_window)


    # Assemble DataFrame
    
    df = pd.concat(
        [
            monthly_returns,
            mom1m,
            mvel.reindex(monthly_returns.index),
            zerotrade.reindex(monthly_returns.index),
            dolvol.reindex(monthly_returns.index),
            chmom,
            ill.reindex(monthly_returns.index),
            retvol.reindex(monthly_returns.index),
            mom6m,
            indmom,
            mom12m,
            mom36m,
            idiovol.reindex(monthly_returns.index),
        ],
        axis=1,
    )

    df.index.name = "date"

    
    # Trim to requested date range and drop pre-IPO NaN rows
    df['returns'] = df['returns'].shift(-1)
    df = df.loc[start_date:end_date]
    df = df[df["returns"].notna()]

    return df


# ── Sequence helpers ─────────────────────────────────────────────────────────
def get_seqs(df, seq_len):
    if len(df) <= seq_len:
        return [], []  # too short for even one sequence — caller skips gracefully
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    xraw = df[["mom1m", "mvel", "zerotrade", "dolvol", "indmom", "chmom",
              "ill", "retvol", "mom6m", "mom12m", "mom36m", "idiovol"]]
    
    yraw = df["returns"]
    
    output = []
    for t in range(seq_len, len(xraw)):
        output.append(xraw[t-seq_len:t].values)

        
    return output, [[y] for y in yraw[seq_len:]]

# ── Normalisation helpers ────────────────────────────────────────────────────
def z_score_norm(series):
    global param_dict
    mu, sigma = param_dict[series.name]
    
    if sigma < 1e-8:
        return pd.Series(np.zeros(len(series)), index=series.index)
    else:
        return (series - mu) / sigma
