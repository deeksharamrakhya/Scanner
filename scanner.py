"""
scanner.py
Batch scanner: runs pattern detection across a watchlist (or index
constituent list) and surfaces only tickers where at least one pattern
(Consolidation, VCP, Rocket Base) was flagged.

Usage:
    python3 scanner.py tickers.txt
    python3 scanner.py RELIANCE.NS TCS.NS INFY.NS AAPL MSFT

    tickers.txt format: one ticker per line, '#' for comments/blank lines ignored.
"""

import sys
import time
import argparse
from analysis import analyze_stock, summarize
from patterns import PATTERN_REGISTRY


def load_watchlist(path: str) -> list:
    tickers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers


def scan(tickers: list, period: str = "1y", pattern_filter: list = None,
          include_fundamentals: bool = False, pause: float = 0.0, verbose: bool = True) -> dict:
    """
    Scan a list of tickers and return only those with at least one pattern hit.

    pattern_filter: optional list like ["VCP", "Rocket Base"] to restrict which
                     patterns count as a "hit". Defaults to all three.
    """
    hits = []
    errors = []
    scanned = 0

    for ticker in tickers:
        try:
            result = analyze_stock(ticker, period=period, include_fundamentals=include_fundamentals,
                                     patterns=pattern_filter)
            scanned += 1
            detected = result["patterns_detected"]
            if detected:
                hits.append(result)
                if verbose:
                    print(f"[HIT] {ticker}: {', '.join(detected)}")
            elif verbose:
                print(f"[ -- ] {ticker}: no pattern")
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            if verbose:
                print(f"[ERR ] {ticker}: {e}")
        if pause:
            time.sleep(pause)

    return {
        "scanned": scanned,
        "total_requested": len(tickers),
        "hits": hits,
        "errors": errors,
    }


def print_report(scan_result: dict):
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE: {scan_result['scanned']}/{scan_result['total_requested']} tickers processed")
    print(f"Hits: {len(scan_result['hits'])} | Errors: {len(scan_result['errors'])}")
    print(f"{'='*60}")

    if not scan_result["hits"]:
        print("No pattern matches found in this watchlist.")
        return

    for result in scan_result["hits"]:
        print("\n" + summarize(result))

    if scan_result["errors"]:
        print(f"\n--- Skipped due to errors ---")
        for e in scan_result["errors"]:
            print(f"  {e['ticker']}: {e['error']}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Scan a watchlist or index for chart pattern hits.",
        epilog=f"Available patterns: {', '.join(PATTERN_REGISTRY.keys())}"
    )
    p.add_argument("tickers", nargs="*", help="Tickers, or a single .txt watchlist file")
    p.add_argument("--index", choices=["nifty50", "sp500"], help="Scan a standard index instead of typing tickers")
    p.add_argument("--pattern", help="Comma-separated pattern names to filter on, e.g. 'VCP,Rocket Base'")
    p.add_argument("--period", default="1y", help="History window to fetch (default: 1y)")
    p.add_argument("--fundamentals", action="store_true", help="Include fundamentals in output (slower)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.index:
        from indices import get_index_constituents
        tickers = get_index_constituents(args.index)
        print(f"Loaded {len(tickers)} tickers from {args.index}")
    elif len(args.tickers) == 1 and args.tickers[0].endswith(".txt"):
        tickers = load_watchlist(args.tickers[0])
    elif args.tickers:
        tickers = args.tickers
    else:
        print("Usage: python3 scanner.py <tickers.txt | TICKER1 TICKER2 ...> [--pattern VCP,...] [--index nifty50|sp500]")
        sys.exit(1)

    pattern_filter = [p.strip() for p in args.pattern.split(",")] if args.pattern else None
    if pattern_filter:
        unknown = [p for p in pattern_filter if p not in PATTERN_REGISTRY]
        if unknown:
            print(f"Warning: unknown pattern name(s) {unknown}. Valid options: {list(PATTERN_REGISTRY.keys())}")

    result = scan(tickers, period=args.period, pattern_filter=pattern_filter,
                  include_fundamentals=args.fundamentals, pause=0.3)
    print_report(result)
