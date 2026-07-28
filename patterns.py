"""
patterns.py
Chart pattern detection: Consolidation, VCP (Volatility Contraction Pattern),
and Rocket Base (tight base after a sharp uptrend, near highs).

These are rule-based heuristics (not ML), designed to flag CANDIDATES for
visual confirmation -- the same approach used by tools like TrendSpider/Chartmill.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data. For Indian stocks, pass ticker with .NS (NSE) or .BO (BSE)
    suffix, e.g. 'RELIANCE.NS'. For US stocks, use the plain ticker, e.g. 'AAPL'.
    """
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_indices(idx_array, min_gap: int):
    """Merge clusters of indices closer than min_gap (e.g. from flat plateaus at
    a single peak/trough) into one representative index per cluster."""
    if len(idx_array) == 0:
        return idx_array
    clusters = [[idx_array[0]]]
    for i in idx_array[1:]:
        if i - clusters[-1][-1] <= min_gap:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return np.array([int(np.mean(c)) for c in clusters])


def _swing_points(series: pd.Series, order: int = 5):
    """Return indices of local maxima and minima, deduplicated across plateaus."""
    highs_idx = argrelextrema(series.values, np.greater_equal, order=order)[0]
    lows_idx = argrelextrema(series.values, np.less_equal, order=order)[0]
    highs_idx = _dedup_indices(highs_idx, min_gap=order)
    lows_idx = _dedup_indices(lows_idx, min_gap=order)
    return highs_idx, lows_idx


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ---------------------------------------------------------------------------
# Pattern 1: Consolidation
# ---------------------------------------------------------------------------

def detect_consolidation(df: pd.DataFrame, lookback: int = 20, threshold_pct: float = 0.5):
    """
    Flags consolidation when the recent trading range (as % of price) is
    meaningfully tighter than the stock's own recent historical average range.
    """
    recent = df.tail(lookback)
    price_range_pct = (recent["high"].max() - recent["low"].min()) / recent["close"].iloc[-1] * 100

    hist_window = df.tail(lookback * 4)
    rolling_ranges = []
    for i in range(lookback, len(hist_window)):
        seg = hist_window.iloc[i - lookback:i]
        r = (seg["high"].max() - seg["low"].min()) / seg["close"].iloc[-1] * 100
        rolling_ranges.append(r)
    avg_hist_range = np.mean(rolling_ranges) if rolling_ranges else price_range_pct

    is_consolidating = price_range_pct < avg_hist_range * threshold_pct

    return {
        "pattern": "Consolidation",
        "detected": bool(is_consolidating),
        "current_range_pct": round(price_range_pct, 2),
        "avg_historical_range_pct": round(avg_hist_range, 2),
        "tightness_ratio": round(price_range_pct / avg_hist_range, 2) if avg_hist_range else None,
    }


# ---------------------------------------------------------------------------
# Pattern 2: VCP (Volatility Contraction Pattern)
# ---------------------------------------------------------------------------

def detect_vcp(df: pd.DataFrame, order: int = 5, min_contractions: int = 2, lookback: int = 120):
    """
    Looks for a series of progressively smaller pullbacks (contractions) from
    swing highs to swing lows, near a prior high, with declining volume.
    Minervini-style: contractions should generally tighten (e.g. 25% -> 15% -> 8%).
    """
    recent = df.tail(lookback).reset_index(drop=True)
    highs_idx, lows_idx = _swing_points(recent["close"], order=order)

    if len(highs_idx) < min_contractions or len(lows_idx) < min_contractions:
        return {"pattern": "VCP", "detected": False, "reason": "Not enough swing points found"}

    # Pair each high with the next low to compute pullback depth
    contractions = []
    all_points = sorted([(i, "H") for i in highs_idx] + [(i, "L") for i in lows_idx])
    for j in range(len(all_points) - 1):
        idx1, type1 = all_points[j]
        idx2, type2 = all_points[j + 1]
        if type1 == "H" and type2 == "L":
            high_price = recent["close"].iloc[idx1]
            low_price = recent["close"].iloc[idx2]
            depth_pct = (high_price - low_price) / high_price * 100
            contractions.append(round(depth_pct, 2))

    if len(contractions) < min_contractions:
        return {"pattern": "VCP", "detected": False, "reason": "Fewer than minimum contractions"}

    last_n = contractions[-min_contractions:]
    is_tightening = all(last_n[i] > last_n[i + 1] for i in range(len(last_n) - 1))

    # Volume trend over the same window should be declining
    vol_recent = recent["volume"].tail(20).mean()
    vol_prior = recent["volume"].iloc[-60:-20].mean() if len(recent) >= 60 else recent["volume"].mean()
    volume_declining = vol_recent < vol_prior

    # Near prior high (within 15%)
    near_high = recent["close"].iloc[-1] >= recent["close"].max() * 0.85

    detected = is_tightening and volume_declining and near_high

    return {
        "pattern": "VCP",
        "detected": bool(detected),
        "contractions_pct": contractions[-4:],
        "tightening": bool(is_tightening),
        "volume_declining": bool(volume_declining),
        "near_prior_high": bool(near_high),
    }


# ---------------------------------------------------------------------------
# Pattern 3: Rocket Base (sharp uptrend -> tight base near highs)
# ---------------------------------------------------------------------------

def detect_rocket_base(df: pd.DataFrame, uptrend_lookback: int = 60, base_lookback: int = 15,
                        min_uptrend_gain_pct: float = 25.0, max_base_range_pct: float = 12.0):
    """
    Detects: (1) a sharp prior uptrend, (2) followed by a tight sideways base
    near the highs, (3) with volume drying up in the base -- a classic
    pre-breakout 'rocket base' setup.
    """
    if len(df) < uptrend_lookback + base_lookback:
        return {"pattern": "Rocket Base", "detected": False, "reason": "Not enough data"}

    pre_base = df.iloc[-(uptrend_lookback + base_lookback):-base_lookback]
    base = df.tail(base_lookback)

    uptrend_gain_pct = (pre_base["close"].iloc[-1] - pre_base["close"].iloc[0]) / pre_base["close"].iloc[0] * 100
    sharp_uptrend = uptrend_gain_pct >= min_uptrend_gain_pct

    base_range_pct = (base["high"].max() - base["low"].min()) / base["close"].iloc[-1] * 100
    tight_base = base_range_pct <= max_base_range_pct

    near_highs = base["close"].iloc[-1] >= df["close"].tail(uptrend_lookback + base_lookback).max() * 0.90

    vol_base = base["volume"].mean()
    vol_uptrend = pre_base["volume"].mean()
    volume_dryup = vol_base < vol_uptrend

    detected = sharp_uptrend and tight_base and near_highs and volume_dryup

    return {
        "pattern": "Rocket Base",
        "detected": bool(detected),
        "prior_uptrend_gain_pct": round(uptrend_gain_pct, 2),
        "base_range_pct": round(base_range_pct, 2),
        "near_highs": bool(near_highs),
        "volume_dryup": bool(volume_dryup),
    }


# ---------------------------------------------------------------------------
# Pattern 4: Cup and Handle
# ---------------------------------------------------------------------------

def detect_cup_and_handle(df: pd.DataFrame, lookback: int = 150, handle_lookback: int = 15,
                           cup_min_depth_pct: float = 12.0, cup_max_depth_pct: float = 50.0,
                           rim_tolerance_pct: float = 6.0, handle_max_depth_pct: float = 15.0):
    """
    Cup: rounded decline-and-recovery back near the prior high.
    Handle: a shallow pullback right after the cup, near the highs, on lighter volume.
    """
    if len(df) < lookback:
        return {"pattern": "Cup and Handle", "detected": False, "reason": "Not enough data"}

    recent = df.tail(lookback)
    cup_zone = recent.iloc[:-handle_lookback]
    handle_zone = recent.tail(handle_lookback)

    third = max(len(cup_zone) // 3, 1)
    left_rim = cup_zone["close"].iloc[:third].max()
    cup_bottom = cup_zone["close"].iloc[third:2 * third].min()
    right_rim = cup_zone["close"].iloc[2 * third:].max()

    cup_depth_pct = (left_rim - cup_bottom) / left_rim * 100
    rims_similar = abs(left_rim - right_rim) / left_rim * 100 <= rim_tolerance_pct
    valid_depth = cup_min_depth_pct <= cup_depth_pct <= cup_max_depth_pct

    handle_high = handle_zone["close"].max()
    handle_low = handle_zone["close"].min()
    handle_depth_pct = (handle_high - handle_low) / handle_high * 100 if handle_high else 0
    shallow_handle = 0 < handle_depth_pct <= handle_max_depth_pct

    vol_handle = handle_zone["volume"].mean()
    vol_cup = cup_zone["volume"].mean()
    volume_dryup = vol_handle < vol_cup

    detected = valid_depth and rims_similar and shallow_handle and volume_dryup

    return {
        "pattern": "Cup and Handle",
        "detected": bool(detected),
        "cup_depth_pct": round(cup_depth_pct, 2),
        "rims_similar": bool(rims_similar),
        "handle_depth_pct": round(handle_depth_pct, 2),
        "volume_dryup_in_handle": bool(volume_dryup),
    }


# ---------------------------------------------------------------------------
# Pattern 5 & 6: Double Top / Double Bottom
# ---------------------------------------------------------------------------

def detect_double_top(df: pd.DataFrame, order: int = 5, lookback: int = 120,
                       peak_tolerance_pct: float = 3.0, min_pullback_pct: float = 5.0):
    """Two comparable peaks with a meaningful trough between them, price now rolling over."""
    recent = df.tail(lookback).reset_index(drop=True)
    highs_idx, lows_idx = _swing_points(recent["close"], order=order)

    if len(highs_idx) < 2:
        return {"pattern": "Double Top", "detected": False, "reason": "Not enough swing highs"}

    i1, i2 = highs_idx[-2], highs_idx[-1]
    h1, h2 = recent["close"].iloc[i1], recent["close"].iloc[i2]
    peaks_similar = abs(h1 - h2) / h1 * 100 <= peak_tolerance_pct

    between_lows = [i for i in lows_idx if i1 < i < i2]
    if not between_lows:
        return {"pattern": "Double Top", "detected": False, "reason": "No trough found between peaks"}
    trough = recent["close"].iloc[between_lows].min()
    pullback_pct = (h1 - trough) / h1 * 100
    sufficient_pullback = pullback_pct >= min_pullback_pct

    rolling_over = recent["close"].iloc[-1] < trough

    detected = peaks_similar and sufficient_pullback and rolling_over

    return {
        "pattern": "Double Top",
        "detected": bool(detected),
        "peak_diff_pct": round(abs(h1 - h2) / h1 * 100, 2),
        "pullback_from_peak_pct": round(pullback_pct, 2),
        "breaking_below_trough": bool(rolling_over),
    }


def detect_double_bottom(df: pd.DataFrame, order: int = 5, lookback: int = 120,
                          trough_tolerance_pct: float = 3.0, min_rally_pct: float = 5.0):
    """Two comparable troughs with a meaningful rally between them, price now breaking up."""
    recent = df.tail(lookback).reset_index(drop=True)
    highs_idx, lows_idx = _swing_points(recent["close"], order=order)

    if len(lows_idx) < 2:
        return {"pattern": "Double Bottom", "detected": False, "reason": "Not enough swing lows"}

    i1, i2 = lows_idx[-2], lows_idx[-1]
    l1, l2 = recent["close"].iloc[i1], recent["close"].iloc[i2]
    troughs_similar = abs(l1 - l2) / l1 * 100 <= trough_tolerance_pct

    between_highs = [i for i in highs_idx if i1 < i < i2]
    if not between_highs:
        return {"pattern": "Double Bottom", "detected": False, "reason": "No peak found between troughs"}
    peak = recent["close"].iloc[between_highs].max()
    rally_pct = (peak - l1) / l1 * 100
    sufficient_rally = rally_pct >= min_rally_pct

    breaking_up = recent["close"].iloc[-1] > peak

    detected = troughs_similar and sufficient_rally and breaking_up

    return {
        "pattern": "Double Bottom",
        "detected": bool(detected),
        "trough_diff_pct": round(abs(l1 - l2) / l1 * 100, 2),
        "rally_from_trough_pct": round(rally_pct, 2),
        "breaking_above_peak": bool(breaking_up),
    }


# ---------------------------------------------------------------------------
# Pattern 7 & 8: Head and Shoulders / Inverse Head and Shoulders
# ---------------------------------------------------------------------------

def detect_head_and_shoulders(df: pd.DataFrame, order: int = 5, lookback: int = 150,
                               shoulder_tolerance_pct: float = 6.0, head_min_prominence_pct: float = 3.0):
    """Bearish reversal: left shoulder, higher head, right shoulder of similar height."""
    recent = df.tail(lookback).reset_index(drop=True)
    highs_idx, lows_idx = _swing_points(recent["close"], order=order)

    if len(highs_idx) < 3:
        return {"pattern": "Head and Shoulders", "detected": False, "reason": "Not enough swing highs"}

    ls_i, h_i, rs_i = highs_idx[-3], highs_idx[-2], highs_idx[-1]
    ls, head, rs = recent["close"].iloc[ls_i], recent["close"].iloc[h_i], recent["close"].iloc[rs_i]

    shoulders_similar = abs(ls - rs) / ((ls + rs) / 2) * 100 <= shoulder_tolerance_pct
    head_prominent = head > ls * (1 + head_min_prominence_pct / 100) and head > rs * (1 + head_min_prominence_pct / 100)

    neckline_lows = [recent["close"].iloc[i] for i in lows_idx if ls_i < i < rs_i]
    neckline = np.mean(neckline_lows) if neckline_lows else None
    breaking_neckline = neckline is not None and recent["close"].iloc[-1] < neckline

    detected = shoulders_similar and head_prominent and bool(breaking_neckline)

    return {
        "pattern": "Head and Shoulders",
        "detected": bool(detected),
        "shoulders_similar": bool(shoulders_similar),
        "head_prominent": bool(head_prominent),
        "neckline": round(float(neckline), 2) if neckline else None,
        "breaking_neckline": bool(breaking_neckline),
    }


def detect_inverse_head_and_shoulders(df: pd.DataFrame, order: int = 5, lookback: int = 150,
                                       shoulder_tolerance_pct: float = 6.0, head_min_prominence_pct: float = 3.0):
    """Bullish reversal: left shoulder, lower head, right shoulder of similar depth."""
    recent = df.tail(lookback).reset_index(drop=True)
    highs_idx, lows_idx = _swing_points(recent["close"], order=order)

    if len(lows_idx) < 3:
        return {"pattern": "Inverse Head and Shoulders", "detected": False, "reason": "Not enough swing lows"}

    ls_i, h_i, rs_i = lows_idx[-3], lows_idx[-2], lows_idx[-1]
    ls, head, rs = recent["close"].iloc[ls_i], recent["close"].iloc[h_i], recent["close"].iloc[rs_i]

    shoulders_similar = abs(ls - rs) / ((ls + rs) / 2) * 100 <= shoulder_tolerance_pct
    head_prominent = head < ls * (1 - head_min_prominence_pct / 100) and head < rs * (1 - head_min_prominence_pct / 100)

    neckline_highs = [recent["close"].iloc[i] for i in highs_idx if ls_i < i < rs_i]
    neckline = np.mean(neckline_highs) if neckline_highs else None
    breaking_neckline = neckline is not None and recent["close"].iloc[-1] > neckline

    detected = shoulders_similar and head_prominent and bool(breaking_neckline)

    return {
        "pattern": "Inverse Head and Shoulders",
        "detected": bool(detected),
        "shoulders_similar": bool(shoulders_similar),
        "head_prominent": bool(head_prominent),
        "neckline": round(float(neckline), 2) if neckline else None,
        "breaking_neckline": bool(breaking_neckline),
    }


# ---------------------------------------------------------------------------
# Pattern 9 & 10: Ascending / Descending Triangle
# ---------------------------------------------------------------------------

def detect_ascending_triangle(df: pd.DataFrame, order: int = 5, lookback: int = 90,
                               flat_tolerance_pct: float = 2.5, min_touches: int = 2):
    """Flat resistance at the top with a series of rising lows -- bullish continuation setup."""
    recent = df.tail(lookback).reset_index(drop=True)
    highs_idx, lows_idx = _swing_points(recent["close"], order=order)

    if len(highs_idx) < min_touches or len(lows_idx) < min_touches:
        return {"pattern": "Ascending Triangle", "detected": False, "reason": "Not enough swing points"}

    last_highs = recent["close"].iloc[highs_idx[-min_touches:]]
    flat_resistance = (last_highs.max() - last_highs.min()) / last_highs.mean() * 100 <= flat_tolerance_pct

    last_lows = recent["close"].iloc[lows_idx[-min_touches:]].values
    rising_lows = all(last_lows[i] < last_lows[i + 1] for i in range(len(last_lows) - 1))

    detected = bool(flat_resistance and rising_lows)

    return {
        "pattern": "Ascending Triangle",
        "detected": detected,
        "resistance_level": round(float(last_highs.mean()), 2),
        "flat_resistance": bool(flat_resistance),
        "rising_lows": bool(rising_lows),
    }


def detect_descending_triangle(df: pd.DataFrame, order: int = 5, lookback: int = 90,
                                flat_tolerance_pct: float = 2.5, min_touches: int = 2):
    """Flat support at the bottom with a series of falling highs -- bearish continuation setup."""
    recent = df.tail(lookback).reset_index(drop=True)
    highs_idx, lows_idx = _swing_points(recent["close"], order=order)

    if len(highs_idx) < min_touches or len(lows_idx) < min_touches:
        return {"pattern": "Descending Triangle", "detected": False, "reason": "Not enough swing points"}

    last_lows = recent["close"].iloc[lows_idx[-min_touches:]]
    flat_support = (last_lows.max() - last_lows.min()) / last_lows.mean() * 100 <= flat_tolerance_pct

    last_highs = recent["close"].iloc[highs_idx[-min_touches:]].values
    falling_highs = all(last_highs[i] > last_highs[i + 1] for i in range(len(last_highs) - 1))

    detected = bool(flat_support and falling_highs)

    return {
        "pattern": "Descending Triangle",
        "detected": detected,
        "support_level": round(float(last_lows.mean()), 2),
        "flat_support": bool(flat_support),
        "falling_highs": bool(falling_highs),
    }


# ---------------------------------------------------------------------------
# Registry -- single source of truth for "what patterns exist"
# Used by analysis.py (to run all of them) and scanner.py (to filter by name)
# ---------------------------------------------------------------------------

PATTERN_REGISTRY = {
    "Consolidation": detect_consolidation,
    "VCP": detect_vcp,
    "Rocket Base": detect_rocket_base,
    "Cup and Handle": detect_cup_and_handle,
    "Double Top": detect_double_top,
    "Double Bottom": detect_double_bottom,
    "Head and Shoulders": detect_head_and_shoulders,
    "Inverse Head and Shoulders": detect_inverse_head_and_shoulders,
    "Ascending Triangle": detect_ascending_triangle,
    "Descending Triangle": detect_descending_triangle,
}


# ---------------------------------------------------------------------------
# Combined scanner
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, period: str = "1y") -> dict:
    df = fetch_ohlcv(ticker, period=period)
    return {
        "ticker": ticker,
        "last_close": round(float(df["close"].iloc[-1]), 2),
        "as_of": str(df.index[-1].date()),
        "consolidation": detect_consolidation(df),
        "vcp": detect_vcp(df),
        "rocket_base": detect_rocket_base(df),
    }


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["RELIANCE.NS", "AAPL"]
    for t in tickers:
        try:
            result = analyze_ticker(t)
            print(f"\n=== {t} (as of {result['as_of']}, close: {result['last_close']}) ===")
            for key in ["consolidation", "vcp", "rocket_base"]:
                print(f"  {key}: {result[key]}")
        except Exception as e:
            print(f"\n=== {t} === ERROR: {e}")
