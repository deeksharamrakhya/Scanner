"""
analysis.py
Stock analysis module: fetches fundamentals + runs pattern detection
(Consolidation, VCP, Rocket Base) automatically for any ticker.

Works for both Indian (NSE: 'RELIANCE.NS', BSE: 'RELIANCE.BO') and
US tickers (e.g. 'AAPL').
"""

from patterns import fetch_ohlcv, PATTERN_REGISTRY


def _fmt_pct(x):
    return f"{x:.2f}%" if x is not None else "N/A"


def get_fundamentals(ticker: str) -> dict:
    """Pull basic fundamental snapshot. Falls back gracefully if fields are missing."""
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "currency": info.get("currency"),
    }


def analyze_stock(ticker: str, period: str = "1y", include_fundamentals: bool = True,
                   patterns: list = None, interval: str = "1d") -> dict:
    """
    Full analysis for a single ticker: price snapshot, fundamentals,
    and pattern detectors run automatically.

    patterns: optional list of pattern names to run (see patterns.PATTERN_REGISTRY
              for the full list). Defaults to running ALL registered patterns.
    interval: bar size passed to yfinance ('1d','1wk','1h','15m', ...).
    """
    df = fetch_ohlcv(ticker, period=period, interval=interval)

    names_to_run = patterns if patterns else list(PATTERN_REGISTRY.keys())
    pattern_results = {}
    for name in names_to_run:
        detector_fn = PATTERN_REGISTRY.get(name)
        if detector_fn is None:
            continue
        pattern_results[name] = detector_fn(df)

    result = {
        "ticker": ticker,
        "as_of": str(df.index[-1].date()),
        "last_close": round(float(df["close"].iloc[-1]), 2),
        "patterns": pattern_results,
    }

    result["patterns_detected"] = [
        p["pattern"] for p in result["patterns"].values() if p.get("detected")
    ]

    if include_fundamentals:
        try:
            result["fundamentals"] = get_fundamentals(ticker)
        except Exception as e:
            result["fundamentals"] = {"error": str(e)}

    return result


def summarize(result: dict) -> str:
    """Plain-English one-block summary of an analyze_stock() result."""
    lines = []
    f = result.get("fundamentals", {})
    name = f.get("name", result["ticker"])
    lines.append(f"{name} ({result['ticker']}) — as of {result['as_of']}, last close {result['last_close']}")

    if f and "error" not in f:
        pe = f.get("pe_ratio")
        mcap = f.get("market_cap")
        lines.append(
            f"  Sector: {f.get('sector', 'N/A')} | PE: {round(pe,1) if pe else 'N/A'} | "
            f"52w range: {f.get('52w_low','N/A')}–{f.get('52w_high','N/A')}"
        )

    detected = result["patterns_detected"]
    if detected:
        lines.append(f"  Patterns flagged: {', '.join(detected)}")
        for p in result["patterns"].values():
            if p.get("detected"):
                details = {k: v for k, v in p.items() if k not in ("pattern", "detected")}
                lines.append(f"    - {p['pattern']}: {details}")
    else:
        lines.append("  No patterns flagged — price action doesn't match any of the tracked setups right now.")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["RELIANCE.NS", "AAPL"]
    for t in tickers:
        try:
            result = analyze_stock(t)
            print("\n" + summarize(result))
        except Exception as e:
            print(f"\n{t}: ERROR — {e}")
