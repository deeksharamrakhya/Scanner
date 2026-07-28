"""
data.py
Thin data-access layer over patterns.fetch_ohlcv, adding:
  - a small on-disk parquet cache (so repeated scans / backtests don't
    re-hammer yfinance for the same ticker+period),
  - ticker normalisation helpers for Indian (NSE/BSE) vs US symbols,
  - a single get_prices() entry point used by the backtest and the UI.

The actual network call still goes through patterns.fetch_ohlcv, so India
(.NS/.BO) vs US handling and the yfinance MultiIndex flattening stay in ONE
place. This module only adds caching + convenience on top.
"""

import os
import time
import hashlib
import pandas as pd

from patterns import fetch_ohlcv

# Cache directory. On Streamlit Cloud / read-only or ephemeral filesystems,
# writing next to the source can fail, so prefer a temp dir and degrade
# gracefully (a failed cache must never break a price fetch).
def _pick_cache_dir():
    import tempfile
    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".price_cache"))
    candidates.append(os.path.join(tempfile.gettempdir(), "scanner_price_cache"))
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            # verify writable
            testf = os.path.join(d, ".w")
            open(testf, "w").close(); os.remove(testf)
            return d
        except Exception:
            continue
    return None  # caching disabled entirely if nothing is writable

_CACHE_DIR = _pick_cache_dir()

# How long a cached file is considered fresh (seconds). Intraday you'd want
# this short; for daily-bar pattern work, a few hours is plenty.
_DEFAULT_TTL = 6 * 3600


def normalize_ticker(ticker: str, market: str = None) -> str:
    """
    Best-effort normalisation.
      - market='india'  -> ensure a .NS suffix if none given
      - market='us'     -> strip any Indian suffix
      - market=None     -> leave as-is (caller already knows the exchange)
    yfinance itself is the source of truth; this only saves the user from
    typing '.NS' every time when they've told us the market.
    """
    t = ticker.strip().upper()
    if market == "india":
        if not (t.endswith(".NS") or t.endswith(".BO")):
            t = f"{t}.NS"
    elif market == "us":
        for suf in (".NS", ".BO"):
            if t.endswith(suf):
                t = t[: -len(suf)]
    return t


def infer_market(ticker: str) -> str:
    """Guess 'india' vs 'us' from the suffix. Used only for display labels."""
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "india"
    return "us"


def _cache_path(ticker: str, period: str, interval: str) -> str:
    key = f"{ticker}|{period}|{interval}".upper()
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    safe = ticker.replace("/", "_").replace("\\", "_")
    return os.path.join(_CACHE_DIR, f"{safe}_{period}_{interval}_{h}.parquet")


def get_prices(ticker: str, period: str = "1y", interval: str = "1d",
               use_cache: bool = True, ttl: int = _DEFAULT_TTL) -> pd.DataFrame:
    """
    Return an OHLCV DataFrame (columns: open/high/low/close/volume, DatetimeIndex).
    Cached to parquet for `ttl` seconds when a writable cache dir exists; if not
    (e.g. read-only cloud fs), it just fetches every time. Never fails over cache.
    """
    caching = use_cache and _CACHE_DIR is not None
    path = _cache_path(ticker, period, interval) if caching else None

    if caching and os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < ttl:
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    return df
            except Exception:
                pass  # corrupt cache -> refetch

    df = fetch_ohlcv(ticker, period=period, interval=interval)

    if caching:
        try:
            df.to_parquet(path)
        except Exception:
            pass  # caching is best-effort; never fail the call over it

    return df


def clear_cache():
    """Wipe the on-disk price cache (no-op if caching is disabled)."""
    if _CACHE_DIR is None:
        return 0
    n = 0
    for f in os.listdir(_CACHE_DIR):
        if f.endswith(".parquet"):
            os.remove(os.path.join(_CACHE_DIR, f))
            n += 1
    return n


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    df = get_prices(t, period="6mo")
    print(f"{t}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    print(df.tail(3))
    # second call should hit cache
    t0 = time.time()
    _ = get_prices(t, period="6mo")
    print(f"cached re-fetch took {time.time()-t0:.3f}s")
