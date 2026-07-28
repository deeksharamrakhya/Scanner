"""
app.py  --  AI Markets Pattern Scanner (Streamlit)

Ties the whole project together:
  - enter tickers or pick an index (Nifty 50 / S&P 500)
  - run the existing scanner / detectors (patterns.PATTERN_REGISTRY)
  - for each hit: annotated price chart + inline walk-forward backtest
  - optional AI commentary (Anthropic API) if ANTHROPIC_API_KEY is set

Run:  streamlit run app.py
Works for both Indian (.NS/.BO) and US tickers via patterns.fetch_ohlcv.
"""

import os
import sys
# Ensure this file's own folder is importable, so `from data import ...` etc.
# work regardless of launch directory. Guarded for envs without __file__.
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import argrelextrema

from data import get_prices, infer_market, normalize_ticker
from analysis import analyze_stock
from patterns import PATTERN_REGISTRY
from backtest import backtest_pattern, PATTERN_DIRECTION
import fundamentals as fund

# ---------------------------------------------------------------------------
# Page + theme
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Pattern Scanner", page_icon="◧", layout="wide")

# --- Soft mint & blush pastel palette (light theme) ---
ACCENT = "#3FB68B"       # muted mint — used for "hit" text, swing lows, up-candles
ACCENT_SOFT = "#A7E8D0"  # pale mint fill
BEAR = "#E88B9A"         # dusty blush — down moves / swing highs
BEAR_SOFT = "#F6C9D2"    # pale blush fill
GRID = "#E8E4EF"         # very soft lavender-grey gridlines
BG = "#FBF9FC"           # near-white with a faint lilac warmth
PANEL = "#F3EFF7"        # soft pastel panel (sidebar, cards)
TEXT = "#4A4658"         # soft dark plum-grey (readable, not harsh black)
MUTED = "#9B95A8"        # muted secondary text

st.markdown(f"""
<style>
  .stApp {{ background:{BG}; color:{TEXT}; }}
  section[data-testid="stSidebar"] {{ background:{PANEL}; }}
  h1,h2,h3,h4 {{ color:{TEXT}; font-family:'DejaVu Sans Mono',monospace; letter-spacing:-.5px; }}
  .mono {{ font-family:'DejaVu Sans Mono',monospace; }}
  .hit {{ color:{ACCENT}; font-weight:600; }}
  .miss {{ color:{MUTED}; }}
  div[data-testid="stMetricValue"] {{ font-family:'DejaVu Sans Mono',monospace; color:{TEXT}; }}
  div[data-testid="stMetricLabel"] {{ color:{MUTED}; }}
  .stDataFrame {{ font-family:'DejaVu Sans Mono',monospace; }}
  /* soften Streamlit's primary button to mint */
  .stButton>button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; }}
  .stButton>button[kind="primary"]:hover {{ background:#35A07A; border-color:#35A07A; }}
</style>
""", unsafe_allow_html=True)

st.title("◧  Pattern Scanner")
st.caption("Rule-based chart-pattern detection + walk-forward backtest · NSE/BSE + US · "
           "candidates for visual confirmation, not signals to trade blindly.")


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def _swings(closes, order=5):
    hi = argrelextrema(closes, np.greater_equal, order=order)[0]
    lo = argrelextrema(closes, np.less_equal, order=order)[0]
    return hi, lo


def price_chart(df: pd.DataFrame, ticker: str, detector_result: dict,
                interval: str = "1d", lookback: int = None):
    """Candlestick + swing-point markers + any level the detector returned
    (neckline / resistance / support), so the chart shows WHY it fired.

    Bar count is interval-aware so candles stay wide enough to read: showing
    500 daily candles makes each one a hairline. Fewer, clearer bars > more."""
    # sensible default number of bars per interval (readable candle width)
    if lookback is None:
        lookback = {"1d": 130, "1wk": 100, "1h": 120, "30m": 100,
                    "15m": 100, "5m": 80, "1m": 60}.get(interval, 130)
    view = df.tail(lookback)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=view.index, open=view["open"], high=view["high"],
        low=view["low"], close=view["close"], name=ticker,
        increasing_line_color=ACCENT, decreasing_line_color=BEAR,
        increasing_fillcolor=ACCENT_SOFT, decreasing_fillcolor=BEAR_SOFT,
        line=dict(width=1),
    ))

    closes = view["close"].values
    hi, lo = _swings(closes, order=5)
    fig.add_trace(go.Scatter(
        x=view.index[hi], y=view["high"].values[hi], mode="markers",
        marker=dict(symbol="triangle-down", size=9, color=BEAR,
                    line=dict(width=0.5, color="#FFFFFF")),
        name="swing high"))
    fig.add_trace(go.Scatter(
        x=view.index[lo], y=view["low"].values[lo], mode="markers",
        marker=dict(symbol="triangle-up", size=9, color=ACCENT,
                    line=dict(width=0.5, color="#FFFFFF")),
        name="swing low"))

    # Draw any horizontal level the detector reported.
    for key, label in [("neckline", "neckline"),
                       ("resistance_level", "resistance"),
                       ("support_level", "support")]:
        lvl = detector_result.get(key)
        if lvl:
            fig.add_hline(y=lvl, line_dash="dot", line_color=MUTED,
                          annotation_text=label, annotation_position="right")

    _label = {"1d": "Daily", "1wk": "Weekly", "1h": "1-Hour",
              "30m": "30-Min", "15m": "15-Min", "5m": "5-Min", "1m": "1-Min"}.get(interval, interval)
    fig.update_layout(
        template="plotly_white", height=480, margin=dict(l=8, r=8, t=34, b=8),
        paper_bgcolor=PANEL, plot_bgcolor="#FFFFFF",
        title=dict(text=f"{ticker} · {_label} · {len(view)} bars", font=dict(size=13, color=TEXT)),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10, color=TEXT)),
        font=dict(family="DejaVu Sans Mono", color=TEXT),
    )
    # Remove non-trading gaps so candles sit flush (weekends always; overnight
    # for intraday). This is the single biggest legibility win.
    breaks = [dict(bounds=["sat", "mon"])]
    if interval in {"1h", "30m", "15m", "5m", "1m"}:
        breaks.append(dict(bounds=[16, 9.25], pattern="hour"))  # rough US session
    fig.update_xaxes(gridcolor=GRID, rangebreaks=breaks)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def equity_chart(trades: list):
    """Compounded per-trade equity curve from a backtest."""
    if not trades:
        return None
    rets = np.array([t["trade_return_pct"] for t in trades]) / 100.0
    eq = np.cumprod(1 + rets)
    fig = go.Figure(go.Scatter(
        y=eq, mode="lines+markers", line=dict(color=ACCENT, width=2.5),
        marker=dict(size=4, color=ACCENT),
        fill="tozeroy", fillcolor="rgba(167,232,208,0.25)"))
    fig.add_hline(y=1.0, line_dash="dot", line_color=MUTED)
    fig.update_layout(
        template="plotly_white", height=220,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor=PANEL, plot_bgcolor="#FFFFFF",
        title=dict(text="equity by trade (1.0 = flat)", font=dict(size=12, color=TEXT)),
        font=dict(family="DejaVu Sans Mono", color=TEXT),
    )
    fig.update_xaxes(gridcolor=GRID, title="trade #")
    fig.update_yaxes(gridcolor=GRID)
    return fig


# ---------------------------------------------------------------------------
# Optional AI commentary
# ---------------------------------------------------------------------------

def ai_commentary(ticker: str, detected: list, bt_stats: dict) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            f"You are a terse equity-desk analyst. Ticker {ticker}. "
            f"Detected chart patterns: {', '.join(detected)}. "
            f"Backtest of the primary pattern on this name: {bt_stats}. "
            "In 3-4 sentences, give a sober read: what the setup implies, and "
            "explicitly caveat the backtest sample size if n_trades is small. "
            "No hype, no financial-advice disclaimers."
        )
        msg = client.messages.create(
            model="claude-sonnet-5", max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as e:
        return f"_AI commentary unavailable: {e}_"


def _kv_table(d: dict):
    """Render a flat dict as a 2-col table."""
    df = pd.DataFrame({"": list(d.keys()), " ": list(d.values())})
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_fundamentals(ticker: str, universe: list):
    """Six expandable panels of yfinance fundamentals for one ticker."""
    st.markdown("**fundamentals**")

    with st.expander("Overview"):
        try:
            ov = fund.overview(ticker)
            summary = ov.pop("Summary", "—")
            _kv_table(ov)
            if summary and summary != "—":
                st.caption(summary[:600] + ("…" if len(summary) > 600 else ""))
        except Exception as e:
            st.info(f"Overview unavailable: {e}")

    with st.expander("Financial position (balance sheet)"):
        try:
            bs = fund.balance_sheet(ticker)
            if bs.empty:
                st.info("No balance-sheet data from yfinance for this ticker.")
            else:
                st.dataframe(bs, use_container_width=True)
        except Exception as e:
            st.info(f"Balance sheet unavailable: {e}")

    with st.expander("P&L / income"):
        try:
            inc = fund.income_statement(ticker)
            if inc.empty:
                st.info("No income-statement data from yfinance for this ticker.")
            else:
                st.dataframe(inc, use_container_width=True)
        except Exception as e:
            st.info(f"Income statement unavailable: {e}")

    with st.expander("Ratios & valuation"):
        try:
            r = fund.ratios(ticker)
            cols = st.columns(len(r))
            for col, (group, vals) in zip(cols, r.items()):
                with col:
                    st.markdown(f"*{group}*")
                    for k, v in vals.items():
                        st.text(f"{k}: {v}")
        except Exception as e:
            st.info(f"Ratios unavailable: {e}")

    with st.expander("Quarterly results"):
        try:
            q = fund.quarterly(ticker)
            if q.empty:
                st.info("No quarterly data from yfinance for this ticker.")
            else:
                st.dataframe(q, use_container_width=True)
        except Exception as e:
            st.info(f"Quarterly unavailable: {e}")

    with st.expander("News & recent developments"):
        try:
            items = fund.news(ticker, limit=8)
            if not items:
                st.info("No recent headlines from yfinance for this ticker. "
                        "(Coverage for smaller .NS names can be thin.)")
            else:
                for it in items:
                    when = f" · {it['when']}" if it["when"] else ""
                    if it["link"]:
                        st.markdown(f"- [{it['title']}]({it['link']})  \n"
                                    f"  <span class='miss'>{it['publisher']}{when}</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"- {it['title']}  \n"
                                    f"  <span class='miss'>{it['publisher']}{when}</span>",
                                    unsafe_allow_html=True)
        except Exception as e:
            st.info(f"News unavailable: {e}")

    with st.expander("Compare (industry peers)"):
        try:
            p = fund.peers(ticker, universe=universe)
            st.caption(f"Sector: {p['sector']} · Industry: {p['industry']}")
            manual = st.text_input(
                "peers (comma-separated) — auto-filled from index matches, edit freely",
                value=", ".join(p["peers"]),
                key=f"peers_{ticker}",
            )
            peer_list = [x.strip() for x in manual.split(",") if x.strip()]
            if p["note"] and not peer_list:
                st.info(p["note"])
            if peer_list:
                tbl = fund.peer_comparison_table([ticker] + peer_list)
                st.dataframe(tbl, use_container_width=True, hide_index=True)
        except Exception as e:
            st.info(f"Peer comparison unavailable: {e}")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("scan setup")
    source = st.radio("universe", ["Type tickers", "Nifty 50", "S&P 500"])

    tickers = []
    if source == "Type tickers":
        raw = st.text_area("tickers (comma or newline)",
                           "RELIANCE.NS, TCS.NS, AAPL, MSFT", height=90)
        tickers = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    else:
        st.caption("Loads constituents live (falls back to a dated list offline).")

    pattern_names = list(PATTERN_REGISTRY.keys())
    chosen = st.multiselect("patterns to scan", pattern_names, default=pattern_names)

    period = st.selectbox("history window", ["1y", "2y", "5y", "6mo"], index=0)
    interval = st.selectbox(
        "bar interval", ["1d", "1wk", "1h", "30m", "15m"], index=0,
        help="Daily is cleanest for chart patterns. Intraday (1h/30m/15m) only "
             "works for short history windows and is flakier on NSE via Colab.",
    )
    horizon = st.slider("backtest holding horizon (bars)", 5, 60, 20, 5)
    bt_period = st.selectbox("backtest history", ["3y", "5y", "10y", "max"], index=1)
    run = st.button("▶  run scan", use_container_width=True, type="primary")

    st.divider()
    st.caption("AI commentary: " +
               ("on" if os.environ.get("ANTHROPIC_API_KEY") else "off (set ANTHROPIC_API_KEY)"))


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

if run:
    if source == "Nifty 50":
        from indices import get_index_constituents
        tickers = get_index_constituents("nifty50")
    elif source == "S&P 500":
        from indices import get_index_constituents
        tickers = get_index_constituents("sp500")

    if not tickers:
        st.warning("No tickers to scan.")
        st.stop()
    if not chosen:
        st.warning("Pick at least one pattern.")
        st.stop()

    # Universe used for peer-matching in the Compare tab. If the user scanned an
    # index this is rich; for a few typed tickers it's thin (peers can be typed
    # manually in the Compare panel).
    scan_universe = list(tickers)

    st.write(f"Scanning **{len(tickers)}** tickers for **{len(chosen)}** patterns …")
    prog = st.progress(0.0)
    results, errors = [], []

    for i, tk in enumerate(tickers):
        try:
            res = analyze_stock(tk, period=period, include_fundamentals=False,
                                patterns=chosen, interval=interval)
            results.append(res)          # keep ALL results, pattern or not
        except Exception as e:
            errors.append((tk, str(e)))
        prog.progress((i + 1) / len(tickers))

    prog.empty()
    hits = [r for r in results if r["patterns_detected"]]
    st.success(f"Done · {len(results)} analysed · {len(hits)} with patterns · {len(errors)} errors")

    # Decide what to render. For a handful of typed tickers, show EVERYTHING
    # (you hand-picked them, you want the full read on each). For a big index
    # scan, showing full chart+backtest+fundamentals for all 50+ would be slow
    # and overwhelming, so show pattern-hits fully and list the rest compactly.
    SHOW_ALL_THRESHOLD = 12
    show_all = len(results) <= SHOW_ALL_THRESHOLD

    if not results:
        st.info("Nothing analysed — all tickers errored. Check symbols / suffixes (.NS for NSE).")
        if errors:
            with st.expander(f"{len(errors)} errors"):
                for tk, e in errors:
                    st.text(f"{tk}: {e}")
        st.stop()

    if show_all:
        to_render = results
        if not hits:
            st.info("No chart patterns matched, but here's the full read on each "
                    "ticker anyway (chart, fundamentals, news).")
    else:
        to_render = hits
        non_hits = [r["ticker"] for r in results if not r["patterns_detected"]]
        if non_hits:
            st.caption(f"Showing {len(hits)} pattern hits in full. "
                       f"{len(non_hits)} scanned with no pattern "
                       f"(type them individually to see their full detail).")
        if not hits:
            st.info("No pattern matches in this large universe right now. "
                    "Type a few specific tickers to see full detail regardless of pattern.")

    for res in to_render:
        tk = res["ticker"]
        detected = res["patterns_detected"]
        mkt = infer_market(tk)
        st.markdown(f"### {tk}  <span class='miss'>· {mkt} · last {res['last_close']}</span>",
                    unsafe_allow_html=True)
        if detected:
            st.markdown("patterns: " + " ".join(f"<span class='hit'>{d}</span>" for d in detected),
                        unsafe_allow_html=True)
            primary = detected[0]
            primary_result = res["patterns"][primary]
        else:
            st.markdown("<span class='miss'>no chart pattern detected — showing "
                        "price action, fundamentals & news</span>", unsafe_allow_html=True)
            primary = None
            primary_result = {}

        # Backtest always runs on DAILY bars over a long window -- intraday
        # backtests over years aren't available and would muddy the stats.
        try:
            df_bt = get_prices(tk, period=bt_period, interval="1d")
        except Exception:
            df_bt = get_prices(tk, period=period, interval="1d")

        # Chart honors the chosen interval. Intraday needs a short period, so
        # clamp it (yfinance rejects e.g. 15m over 5y).
        _INTRADAY = {"1h", "30m", "15m", "5m", "1m"}
        chart_period = period
        if interval in _INTRADAY:
            chart_period = "1mo" if interval in {"15m", "5m", "1m"} else "3mo"
        try:
            df_chart = get_prices(tk, period=chart_period, interval=interval)
        except Exception:
            df_chart = df_bt  # fall back to daily if intraday fetch fails

        col_chart, col_stats = st.columns([3, 2])
        bt = None  # holds backtest result if a pattern fired; stays None otherwise
        with col_chart:
            st.plotly_chart(price_chart(df_chart, tk, primary_result, interval=interval),
                            use_container_width=True,
                            config={"scrollZoom": True, "displayModeBar": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToAdd": ["drawline", "eraseshape"]})
        with col_stats:
            if primary is None:
                st.markdown("**no pattern to backtest**")
                st.caption("This ticker didn't match any tracked pattern in the "
                           "current window, so there's no signal to backtest. The "
                           "chart, fundamentals, and news below still apply.")
            else:
                st.markdown(f"**backtest · {primary}** ({PATTERN_DIRECTION.get(primary,'long')})")
                st.caption("backtest on daily bars")
                bt = backtest_pattern(df_bt, primary, horizon=horizon)
                if bt["n_trades"] == 0:
                    st.info(bt.get("note", "No historical signals to backtest."))
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("trades", bt["n_trades"])
                    c2.metric("win rate", f"{bt['win_rate_pct']}%")
                    c3.metric("avg ret", f"{bt['mean_return_pct']}%")
                    c4, c5, c6 = st.columns(3)
                    c4.metric("median", f"{bt['median_return_pct']}%")
                    c5.metric("max DD", f"{bt['max_drawdown_pct']}%")
                    c6.metric("equity", f"{bt['final_equity_x']}x")
                    eqfig = equity_chart(bt["trades"])
                    if eqfig:
                        st.plotly_chart(eqfig, use_container_width=True,
                                        config={"displayModeBar": True, "displaylogo": False})

        with st.expander("detector detail"):
            if primary_result:
                st.json({k: v for k, v in primary_result.items()})
            else:
                st.caption("No pattern fired, so no detector detail.")

        # ---- Fundamentals: one expandable panel per section (yfinance-sourced) ----
        render_fundamentals(tk, scan_universe)

        # AI commentary only when there's a pattern + backtest to talk about
        if primary is not None:
            bt_stats = {}
            if bt and bt.get("n_trades"):
                bt_stats = {k: bt.get(k) for k in ("n_trades", "win_rate_pct", "mean_return_pct")}
            commentary = ai_commentary(tk, detected, bt_stats)
            if commentary:
                st.markdown("**desk read**")
                st.markdown(commentary)

        st.divider()

    if errors:
        with st.expander(f"{len(errors)} errors"):
            for tk, e in errors:
                st.text(f"{tk}: {e}")

else:
    st.info("Configure the scan in the sidebar and hit **run scan**. "
            "Tip: start with a handful of typed tickers before scanning a full index.")
    st.markdown("""
**How to read this**
- Each **hit** shows a candlestick chart with swing highs/lows marked, plus any
  level the detector keyed on (neckline / resistance / support).
- The **backtest** panel replays history bar-by-bar with no lookahead: at each
  past date it asks *"would this pattern have fired then?"* and measures the
  forward return over your chosen horizon. Small `n_trades` = treat with caution.
- Bearish patterns (Double Top, H&S, Descending Triangle) are backtested **short**,
  so a positive avg return still means *"the pattern worked."*
""")
