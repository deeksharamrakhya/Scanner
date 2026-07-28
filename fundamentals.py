"""
fundamentals.py
Legit fundamental data for the detail panel, sourced entirely from yfinance
(company filings), NOT scraped from Screener or any site that pre-compiles data.

Exposes one function per tab:
  overview(t)          -> dict of headline facts
  balance_sheet(t)     -> tidy DataFrame (Financial Position)
  income_statement(t)  -> tidy DataFrame (P&L)
  ratios(t)            -> dict of valuation / profitability ratios
  quarterly(t)         -> tidy DataFrame (recent quarterly results)
  peers(t, universe)   -> list of same-industry tickers drawn from an index list

Everything is best-effort: yfinance coverage for smaller .NS names is patchy,
so each function degrades to partial / empty rather than raising.
"""

import functools
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


@functools.lru_cache(maxsize=128)
def _ticker(t: str):
    return yf.Ticker(t)


@functools.lru_cache(maxsize=128)
def _info(t: str) -> dict:
    try:
        return _ticker(t).info or {}
    except Exception:
        return {}


def _fmt_big(n):
    """Human-format large money numbers (cr / B aware-ish, currency-agnostic)."""
    if n is None:
        return "—"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n/div:.2f}{unit}"
    return f"{n:.0f}"


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def overview(t: str) -> dict:
    info = _info(t)
    return {
        "Name": info.get("longName") or info.get("shortName") or t,
        "Sector": info.get("sector") or "—",
        "Industry": info.get("industry") or "—",
        "Country": info.get("country") or "—",
        "Currency": info.get("currency") or "—",
        "Market cap": _fmt_big(info.get("marketCap")),
        "Current price": info.get("currentPrice") or info.get("previousClose") or "—",
        "52W high": info.get("fiftyTwoWeekHigh") or "—",
        "52W low": info.get("fiftyTwoWeekLow") or "—",
        "Employees": info.get("fullTimeEmployees") or "—",
        "Website": info.get("website") or "—",
        "Summary": (info.get("longBusinessSummary") or "").strip() or "—",
    }


# --------------------------------------------------------------------------- #
# Financial statements -> tidy DataFrames
# --------------------------------------------------------------------------- #
def _tidy_statement(df: pd.DataFrame, rows_wanted=None) -> pd.DataFrame:
    """yfinance returns statements with line-items as the index and period
    columns. Format money values, keep the newest ~4 periods, optionally
    filter to a subset of well-known line items."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # newest 4 periods, columns are Timestamps
    out = out.iloc[:, :4]
    out.columns = [c.date().isoformat() if hasattr(c, "date") else str(c) for c in out.columns]
    if rows_wanted:
        keep = [r for r in rows_wanted if r in out.index]
        if keep:
            out = out.loc[keep]
    # format values
    out = out.applymap(_fmt_big)
    out.index.name = "Line item"
    return out


def balance_sheet(t: str) -> pd.DataFrame:
    try:
        bs = _ticker(t).balance_sheet
    except Exception:
        bs = None
    rows = ["Total Assets", "Total Liabilities Net Minority Interest",
            "Total Equity Gross Minority Interest", "Total Debt",
            "Cash And Cash Equivalents", "Net Debt", "Working Capital",
            "Retained Earnings", "Common Stock Equity"]
    return _tidy_statement(bs, rows)


def income_statement(t: str) -> pd.DataFrame:
    try:
        inc = _ticker(t).financials
    except Exception:
        inc = None
    rows = ["Total Revenue", "Gross Profit", "Operating Income",
            "EBITDA", "EBIT", "Net Income", "Basic EPS", "Diluted EPS",
            "Interest Expense", "Tax Provision"]
    return _tidy_statement(inc, rows)


def quarterly(t: str) -> pd.DataFrame:
    try:
        q = _ticker(t).quarterly_financials
    except Exception:
        q = None
    rows = ["Total Revenue", "Operating Income", "Net Income",
            "Basic EPS", "Diluted EPS", "EBITDA"]
    return _tidy_statement(q, rows)


# --------------------------------------------------------------------------- #
# Ratios
# --------------------------------------------------------------------------- #
def ratios(t: str) -> dict:
    info = _info(t)

    def pct(x):
        return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"

    def num(x, d=2):
        return f"{x:.{d}f}" if isinstance(x, (int, float)) else "—"

    return {
        "Valuation": {
            "P/E (trailing)": num(info.get("trailingPE")),
            "P/E (forward)": num(info.get("forwardPE")),
            "P/B": num(info.get("priceToBook")),
            "P/S": num(info.get("priceToSalesTrailing12Months")),
            "EV/EBITDA": num(info.get("enterpriseToEbitda")),
            "PEG": num(info.get("pegRatio")),
        },
        "Profitability": {
            "ROE": pct(info.get("returnOnEquity")),
            "ROA": pct(info.get("returnOnAssets")),
            "Gross margin": pct(info.get("grossMargins")),
            "Operating margin": pct(info.get("operatingMargins")),
            "Net margin": pct(info.get("profitMargins")),
        },
        "Financial health": {
            "Debt/Equity": num(info.get("debtToEquity")),
            "Current ratio": num(info.get("currentRatio")),
            "Quick ratio": num(info.get("quickRatio")),
            "Dividend yield": pct(info.get("dividendYield")) if info.get("dividendYield") else "—",
        },
        "Growth": {
            "Revenue growth (YoY)": pct(info.get("revenueGrowth")),
            "Earnings growth (YoY)": pct(info.get("earningsGrowth")),
        },
    }


# --------------------------------------------------------------------------- #
# Peers (industry match within a provided universe -- no scraping)
# --------------------------------------------------------------------------- #
def peers(t: str, universe: list = None, max_n: int = 8) -> dict:
    """
    Find same-industry peers by matching yfinance 'industry' across the tickers
    in `universe` (e.g. the Nifty 50 / S&P 500 list already loaded). This is a
    pragmatic peer set, not an exhaustive one -- yfinance has no peer endpoint.
    Returns {'industry':..., 'sector':..., 'peers':[...], 'note':...}.
    """
    info = _info(t)
    industry = info.get("industry")
    sector = info.get("sector")

    result = {"industry": industry or "—", "sector": sector or "—",
              "peers": [], "note": ""}

    if not universe:
        result["note"] = ("No index universe loaded to match against — "
                          "scan an index (Nifty 50 / S&P 500) or add peers manually.")
        return result
    if not industry:
        result["note"] = "yfinance did not return an industry for this ticker."
        return result

    matches = []
    for cand in universe:
        if cand.upper() == t.upper():
            continue
        cinfo = _info(cand)
        if cinfo.get("industry") == industry:
            matches.append(cand)
        elif not matches and cinfo.get("sector") == sector:
            # weaker fallback: same sector, only if we find no industry matches
            pass
        if len(matches) >= max_n:
            break

    result["peers"] = matches
    if not matches:
        result["note"] = (f"No same-industry peers found within the loaded "
                          f"universe for industry '{industry}'.")
    return result


def peer_comparison_table(tickers: list) -> pd.DataFrame:
    """Small side-by-side valuation table for a ticker + its peers."""
    recs = []
    for t in tickers:
        info = _info(t)
        recs.append({
            "Ticker": t,
            "Mkt cap": _fmt_big(info.get("marketCap")),
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "ROE %": (info.get("returnOnEquity") or 0) * 100 if info.get("returnOnEquity") else None,
            "Op margin %": (info.get("operatingMargins") or 0) * 100 if info.get("operatingMargins") else None,
            "Rev growth %": (info.get("revenueGrowth") or 0) * 100 if info.get("revenueGrowth") else None,
        })
    df = pd.DataFrame(recs)
    for c in ["P/E", "P/B", "ROE %", "Op margin %", "Rev growth %"]:
        if c in df:
            df[c] = df[c].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else "—")
    return df


def news(t: str, limit: int = 8) -> list:
    """
    Recent headlines for a ticker via yfinance's built-in news feed (free, no
    key). Returns a list of {title, publisher, link, when}. yfinance's news
    schema has shifted across versions, so we defensively handle both the old
    flat shape and the newer nested {'content': {...}} shape.
    """
    try:
        raw = _ticker(t).news or []
    except Exception:
        return []

    items = []
    for n in raw[:limit]:
        # newer yfinance nests everything under 'content'
        c = n.get("content", n) if isinstance(n, dict) else {}
        title = c.get("title") or n.get("title")
        if not title:
            continue
        # link can live in a few places depending on version
        link = (n.get("link")
                or (c.get("canonicalUrl") or {}).get("url")
                or (c.get("clickThroughUrl") or {}).get("url")
                or "")
        publisher = (n.get("publisher")
                     or (c.get("provider") or {}).get("displayName")
                     or "—")
        when = (c.get("pubDate") or c.get("displayTime")
                or n.get("providerPublishTime") or "")
        # normalise epoch ints to a readable date
        if isinstance(when, (int, float)):
            try:
                when = pd.to_datetime(when, unit="s").strftime("%Y-%m-%d %H:%M")
            except Exception:
                when = str(when)
        elif isinstance(when, str) and "T" in when:
            when = when.split("T")[0]
        items.append({"title": title, "publisher": publisher,
                      "link": link, "when": when})
    return items


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print("OVERVIEW:", overview(t))
    print("\nBALANCE SHEET:\n", balance_sheet(t))
    print("\nRATIOS:", ratios(t))
