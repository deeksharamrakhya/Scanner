"""
indices.py
Fetches index constituent tickers for Nifty 50 and S&P 500, formatted for
yfinance (Nifty 50 tickers get the '.NS' suffix automatically).

IMPORTANT: Index composition changes periodically (reconstitutions, M&A,
delistings). This module tries to fetch the CURRENT list live; if that fails
(no internet, site structure changed, etc.) it falls back to a small,
explicitly-dated hardcoded list. Always treat the fallback as approximate and
verify against the official source before relying on it:
  - Nifty 50 official list: https://www.nseindia.com/index-nifty50
  - S&P 500 official list: https://www.spglobal.com/spdji/en/indices/equity/sp-500/
"""

import pandas as pd


# Fallback only -- NOT guaranteed current. Last manually checked: 2026-07.
# Used only if the live fetch below fails.
_NIFTY50_FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC",
    "LT", "KOTAKBANK", "SBIN", "AXISBANK", "HINDUNILVR", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "WIPRO", "ADANIENT", "ONGC", "NTPC", "POWERGRID", "M&M", "TATAMOTORS",
    "TATASTEEL", "JSWSTEEL", "HCLTECH", "TECHM", "INDUSINDBK", "BAJAJFINSV",
    "DIVISLAB", "DRREDDY", "CIPLA", "GRASIM", "BRITANNIA", "EICHERMOT",
    "HEROMOTOCO", "BAJAJ-AUTO", "COALINDIA", "HINDALCO", "APOLLOHOSP",
    "SBILIFE", "HDFCLIFE", "BPCL", "SHRIRAMFIN", "UPL", "LTIM", "ADANIPORTS",
]

_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA",
    "JPM", "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "NFLX",
]


def get_nifty50_constituents(prefer_live: bool = True) -> list:
    """Returns Nifty 50 tickers with '.NS' suffix, ready for yfinance."""
    tickers = None
    if prefer_live:
        try:
            # NSE's archive endpoint for the official Nifty 50 constituent CSV.
            # Note: nseindia.com sometimes blocks requests without browser-like
            # headers/session cookies -- if this fails, add headers via requests
            # + pass the raw bytes to pd.read_csv, or use nsepython/nsetools.
            url = "https://archive.nseindia.com/content/indices/ind_nifty50list.csv"
            df = pd.read_csv(url)
            tickers = df["Symbol"].str.strip().tolist()
        except Exception:
            tickers = None

    if not tickers:
        print("Note: live Nifty 50 fetch failed -- using fallback list "
              "(may be stale, verify against nseindia.com).")
        tickers = _NIFTY50_FALLBACK

    return [f"{t}.NS" if not t.endswith(".NS") else t for t in tickers]


def get_sp500_constituents(prefer_live: bool = True) -> list:
    """Returns S&P 500 tickers, ready for yfinance."""
    tickers = None
    if prefer_live:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception:
            tickers = None

    if not tickers:
        print("Note: live S&P 500 fetch failed -- using a SHORT fallback list "
              "(top names only, NOT the full 500; verify against spglobal.com).")
        tickers = _SP500_FALLBACK

    return tickers


def get_index_constituents(index_name: str) -> list:
    index_name = index_name.lower().strip()
    if index_name in ("nifty50", "nifty 50", "nifty"):
        return get_nifty50_constituents()
    if index_name in ("sp500", "s&p500", "s&p 500", "spx"):
        return get_sp500_constituents()
    raise ValueError(f"Unknown index '{index_name}'. Supported: 'nifty50', 'sp500'")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "nifty50"
    tickers = get_index_constituents(name)
    print(f"{name}: {len(tickers)} tickers")
    print(tickers)
