"""
backtest.py
Walk-forward backtest for the pattern detectors in patterns.PATTERN_REGISTRY.

WHY WALK-FORWARD:
The detectors in patterns.py only ever judge the MOST RECENT window of the
DataFrame you hand them (they operate on df.tail(lookback)). So to ask
"did this pattern have any edge historically?" we can't call the detector once
on the whole series -- that only tells us about *today*. Instead we replay
history bar by bar: at each date t we feed the detector df.iloc[:t+1], which is
exactly the information a trader would have had AT THAT TIME (no lookahead), and
record a signal whenever it fires. Then we measure the forward return from the
signal bar over a fixed holding horizon.

This is a SIGNAL-QUALITY backtest, not a full portfolio simulator: fixed holding
period, one unit per signal, no position sizing / stops / slippage / costs
(there's a simple cost knob). It answers "when this detector fired, what happened
next, on average?" -- which is the honest first question for any pattern.

Direction:
  bullish patterns  -> simulate LONG  (profit if price rises)
  bearish patterns  -> simulate SHORT (profit if price falls)
Return is always expressed as PnL in the direction of the trade, so a positive
mean_return is "the signal worked" regardless of long/short.
"""

import numpy as np
import pandas as pd

from patterns import PATTERN_REGISTRY, fetch_ohlcv

# Which way each pattern leans. Bearish ones are shorted in the backtest.
PATTERN_DIRECTION = {
    "Consolidation": "long",          # neutral by nature; treated as long-bias breakout
    "VCP": "long",
    "Rocket Base": "long",
    "Cup and Handle": "long",
    "Double Top": "short",
    "Double Bottom": "long",
    "Head and Shoulders": "short",
    "Inverse Head and Shoulders": "long",
    "Ascending Triangle": "long",
    "Descending Triangle": "short",
}


def _forward_return(closes: np.ndarray, entry_idx: int, horizon: int) -> float:
    """Simple close-to-close return from entry_idx to entry_idx+horizon."""
    exit_idx = min(entry_idx + horizon, len(closes) - 1)
    entry = closes[entry_idx]
    exit_ = closes[exit_idx]
    return (exit_ - entry) / entry


def backtest_pattern(df: pd.DataFrame, pattern_name: str,
                     horizon: int = 20, min_history: int = 160,
                     cooldown: int = 20, cost_pct: float = 0.0,
                     step: int = 1) -> dict:
    """
    Replay `df` bar by bar and evaluate one pattern.

    horizon    : bars held after entry (entry is next bar's close).
    min_history: bars required before we start testing (detectors need lookback).
    cooldown   : after a signal, skip this many bars before allowing another,
                 so one persistent setup isn't double-counted every day.
    cost_pct   : round-trip friction subtracted from each trade's return (e.g.
                 0.2 = 20 bps). Rough proxy for brokerage+slippage+STT.
    step       : evaluate every `step` bars (1 = every bar; raise it to speed up).

    Returns a dict of aggregate stats plus the list of individual trades.
    """
    if pattern_name not in PATTERN_REGISTRY:
        raise ValueError(f"Unknown pattern '{pattern_name}'. "
                         f"Options: {list(PATTERN_REGISTRY)}")

    detector = PATTERN_REGISTRY[pattern_name]
    direction = PATTERN_DIRECTION.get(pattern_name, "long")
    sign = 1.0 if direction == "long" else -1.0

    closes = df["close"].values
    dates = df.index
    n = len(df)

    trades = []
    last_signal_i = -10 ** 9

    # We need `horizon` bars AFTER a signal to measure its outcome, so stop early.
    last_testable = n - horizon - 1

    for i in range(min_history, last_testable, step):
        if i - last_signal_i < cooldown:
            continue

        window = df.iloc[: i + 1]          # info available AS OF bar i (no lookahead)
        try:
            res = detector(window)
        except Exception:
            continue

        if not res.get("detected"):
            continue

        # Entry on the NEXT bar's close (you can't act on bar i until it closes).
        entry_idx = i + 1
        raw_ret = _forward_return(closes, entry_idx, horizon)
        trade_ret = sign * raw_ret - cost_pct / 100.0

        trades.append({
            "signal_date": str(dates[i].date()),
            "entry_date": str(dates[entry_idx].date()),
            "entry_price": round(float(closes[entry_idx]), 2),
            "direction": direction,
            "raw_forward_return_pct": round(float(raw_ret) * 100, 2),
            "trade_return_pct": round(float(trade_ret) * 100, 2),
        })
        last_signal_i = i

    return _aggregate(pattern_name, direction, horizon, trades, n)


def _aggregate(pattern_name, direction, horizon, trades, n_bars) -> dict:
    if not trades:
        return {
            "pattern": pattern_name,
            "direction": direction,
            "horizon_bars": horizon,
            "n_trades": 0,
            "note": "No signals fired over this history window.",
            "trades": [],
        }

    rets = np.array([t["trade_return_pct"] for t in trades])
    wins = rets > 0

    # Annualised-ish Sharpe proxy on the per-trade returns (not calendar-scaled;
    # a rough comparability number across patterns, not a real Sharpe).
    sharpe_like = float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0

    # Expectancy per trade and a naive equity curve (compounding each trade).
    equity = np.cumprod(1 + rets / 100.0)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min() * 100)

    return {
        "pattern": pattern_name,
        "direction": direction,
        "horizon_bars": horizon,
        "n_trades": len(trades),
        "win_rate_pct": round(float(wins.mean() * 100), 1),
        "mean_return_pct": round(float(rets.mean()), 2),
        "median_return_pct": round(float(np.median(rets)), 2),
        "best_pct": round(float(rets.max()), 2),
        "worst_pct": round(float(rets.min()), 2),
        "std_pct": round(float(rets.std()), 2),
        "sharpe_like": round(sharpe_like, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_equity_x": round(float(equity[-1]), 3),
        "trades": trades,
    }


def backtest_ticker(ticker: str, pattern_name: str, period: str = "5y",
                    horizon: int = 20, **kwargs) -> dict:
    """Convenience: fetch prices for a ticker, then backtest one pattern on it."""
    # Use the caching layer if available, else fall straight through.
    try:
        from data import get_prices
        df = get_prices(ticker, period=period)
    except Exception:
        df = fetch_ohlcv(ticker, period=period)
    out = backtest_pattern(df, pattern_name, horizon=horizon, **kwargs)
    out["ticker"] = ticker
    out["period"] = period
    return out


def backtest_all_patterns(ticker: str, period: str = "5y", horizon: int = 20,
                          **kwargs) -> pd.DataFrame:
    """Run every registered pattern on one ticker; return a comparison table."""
    try:
        from data import get_prices
        df = get_prices(ticker, period=period)
    except Exception:
        df = fetch_ohlcv(ticker, period=period)

    rows = []
    for name in PATTERN_REGISTRY:
        r = backtest_pattern(df, name, horizon=horizon, **kwargs)
        rows.append({
            "pattern": name,
            "direction": r["direction"],
            "n_trades": r["n_trades"],
            "win_rate_pct": r.get("win_rate_pct"),
            "mean_return_pct": r.get("mean_return_pct"),
            "sharpe_like": r.get("sharpe_like"),
            "max_dd_pct": r.get("max_drawdown_pct"),
            "final_equity_x": r.get("final_equity_x"),
        })
    return pd.DataFrame(rows).sort_values(
        "mean_return_pct", ascending=False, na_position="last"
    ).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    pattern = sys.argv[2] if len(sys.argv) > 2 else "Cup and Handle"
    print(f"Backtesting '{pattern}' on {ticker} ...")
    r = backtest_ticker(ticker, pattern, period="5y", horizon=20)
    for k, v in r.items():
        if k != "trades":
            print(f"  {k}: {v}")
    print(f"  first 3 trades: {r['trades'][:3]}")
