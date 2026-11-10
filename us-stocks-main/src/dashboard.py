import os
import subprocess
import sys
import time

import pandas as pd
import streamlit as st

from live_predict import run_live_predictions


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="US Stock ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_FOLDER = os.path.join(
    BASE_DIR,
    "models"
)

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final"
)

PREDICTION_FILE = os.path.join(
    MODELS_FOLDER,
    "all_stock_predictions.csv"
)

RANKING_FILE = os.path.join(
    MODELS_FOLDER,
    "stock_rankings.csv"
)


# ============================================================
# CUSTOM CSS — MARKET TERMINAL THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── GLOBAL ── */
    :root {
        --bg-primary:    #0a0e17;
        --bg-secondary:  #111827;
        --bg-card:       #141c2b;
        --bg-card-hover: #1a2438;
        --border-dim:    #1e293b;
        --border-glow:   #1e3a5f;
        --text-primary:  #e0e6ed;
        --text-secondary:#8896a6;
        --text-muted:    #4b5c6e;
        --green:         #00e676;
        --green-dim:     #00e67633;
        --red:           #ff1744;
        --red-dim:       #ff174433;
        --amber:         #ffab00;
        --amber-dim:     #ffab0033;
        --cyan:          #00bcd4;
        --blue:          #2979ff;
        --purple:        #7c4dff;
    }

    html, body, [class*="stApp"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── HIDE DEFAULT HEADER ── */
    header[data-testid="stHeader"] {
        background: linear-gradient(180deg, var(--bg-primary) 0%, transparent 100%) !important;
    }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #0a0e17 100%) !important;
        border-right: 1px solid var(--border-dim) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }

    /* ── MARKET HEADER ── */
    .market-header {
        background: linear-gradient(135deg, #0d1321 0%, #111827 50%, #0a1628 100%);
        border: 1px solid var(--border-glow);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }

    .market-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg,
            transparent 0%,
            var(--green) 20%,
            var(--cyan) 50%,
            var(--green) 80%,
            transparent 100%
        );
        opacity: 0.8;
    }

    .market-header::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, var(--green-dim) 0%, transparent 70%);
        pointer-events: none;
    }

    .market-header .brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }

    .market-header .brand-icon {
        font-size: 32px;
        filter: drop-shadow(0 0 8px var(--green-dim));
    }

    .market-header .brand-name {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 0%, var(--green) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .market-header .brand-tag {
        font-size: 13px;
        color: var(--text-secondary);
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 500;
    }

    .market-header .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--green-dim);
        border: 1px solid var(--green);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 11px;
        font-weight: 700;
        color: var(--green);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 10px;
    }

    .market-header .live-badge .pulse {
        width: 8px;
        height: 8px;
        background: var(--green);
        border-radius: 50%;
        animation: pulse-glow 1.5s ease-in-out infinite;
    }

    @keyframes pulse-glow {
        0%, 100% { opacity: 1; box-shadow: 0 0 4px var(--green); }
        50%      { opacity: 0.4; box-shadow: 0 0 12px var(--green); }
    }

    /* ── SECTION TITLES ── */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: var(--text-primary);
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-dim);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-title .icon {
        font-size: 20px;
        filter: drop-shadow(0 0 6px var(--green-dim));
    }

    .section-title .label {
        background: linear-gradient(90deg, var(--text-primary), var(--text-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── METRIC CARDS ── */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--bg-card) 0%, #0f1623 100%);
        border: 1px solid var(--border-dim);
        border-radius: 12px;
        padding: 18px 20px;
        transition: all 0.25s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--border-glow);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stMetric"] label {
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }

    /* ── DATAFRAMES ── */
    [data-testid="stDataFrame"],
    .stDataFrame {
        border: 1px solid var(--border-dim) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.2) !important;
    }

    /* ── EXPANDER ── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-dim) !important;
        border-radius: 12px !important;
        background: var(--bg-card) !important;
    }

    /* ── SIGNAL BADGES ── */
    .signal-up {
        display: inline-block;
        background: var(--green-dim);
        color: var(--green);
        border: 1px solid var(--green);
        border-radius: 6px;
        padding: 4px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .signal-down {
        display: inline-block;
        background: var(--red-dim);
        color: var(--red);
        border: 1px solid var(--red);
        border-radius: 6px;
        padding: 4px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .signal-uncertain {
        display: inline-block;
        background: var(--amber-dim);
        color: var(--amber);
        border: 1px solid var(--amber);
        border-radius: 6px;
        padding: 4px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    /* ── SUMMARY CARD ROW ── */
    .summary-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }

    .summary-card {
        flex: 1;
        background: linear-gradient(145deg, var(--bg-card) 0%, #0f1623 100%);
        border: 1px solid var(--border-dim);
        border-radius: 14px;
        padding: 22px 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .summary-card:hover {
        border-color: var(--border-glow);
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
    }

    .summary-card .card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .summary-card .card-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    .summary-card.card-total  .card-value { color: var(--cyan); }
    .summary-card.card-up     .card-value { color: var(--green); }
    .summary-card.card-down   .card-value { color: var(--red); }
    .summary-card.card-uncertain .card-value { color: var(--amber); }

    .summary-card.card-total  { border-top: 3px solid var(--cyan); }
    .summary-card.card-up     { border-top: 3px solid var(--green); }
    .summary-card.card-down   { border-top: 3px solid var(--red); }
    .summary-card.card-uncertain { border-top: 3px solid var(--amber); }

    /* ── RISK BADGE ── */
    .risk-high   { color: var(--red);   font-weight: 700; }
    .risk-medium { color: var(--amber); font-weight: 700; }
    .risk-low    { color: var(--green); font-weight: 700; }

    /* ── FOOTER ── */
    .market-footer {
        text-align: center;
        padding: 20px 0 10px 0;
        border-top: 1px solid var(--border-dim);
        margin-top: 40px;
        color: var(--text-muted);
        font-size: 12px;
        letter-spacing: 1px;
    }

    .market-footer a {
        color: var(--green);
        text-decoration: none;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-glow);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* ── DIVIDERS ── */
    hr {
        border-color: var(--border-dim) !important;
        opacity: 0.5;
    }

    /* ── CHARTS ── */
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {
        border: 1px solid var(--border-dim);
        border-radius: 12px;
        padding: 8px;
        background: var(--bg-card);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE — MARKET HEADER
# ============================================================

st.markdown(
    """
    <div class="market-header">
        <div class="brand">
            <span class="brand-icon">📊</span>
            <span class="brand-name">US Stock Market ML Dashboard</span>
        </div>
        <div class="brand-tag">
            XGBoost Predictions · Technical Analysis · Risk Metrics · Stock Ranking
        </div>
        <div class="live-badge">
            <span class="pulse"></span>
            Market Intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def format_percentage(value):

    try:

        if pd.isna(value):
            return "N/A"

        value = float(value)

        # Handle values already stored as percentages
        if abs(value) > 1:
            return f"{value:.2f}%"

        return f"{value:.2%}"

    except Exception:

        return "N/A"


def format_number(value, decimals=2):

    try:

        if pd.isna(value):
            return "N/A"

        return f"{float(value):.{decimals}f}"

    except Exception:

        return "N/A"


def get_prediction_file():

    if not os.path.exists(
        PREDICTION_FILE
    ):

        return None

    try:

        df = pd.read_csv(
            PREDICTION_FILE
        )

        df = clean_column_names(df)

        return df

    except Exception as error:

        st.error(
            "Unable to read prediction file:"
        )

        st.code(
            str(error)
        )

        return None


def get_historical_data(ticker):

    file_path = os.path.join(
        DATA_FOLDER,
        f"{ticker}.csv"
    )

    if not os.path.exists(
        file_path
    ):

        return None

    try:

        df = pd.read_csv(
            file_path
        )

        df = clean_column_names(df)

        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df = df.sort_values(
                "Date"
            )

        return df

    except Exception:

        return None


# ============================================================
# LOAD PREDICTIONS
# ============================================================

predictions = get_prediction_file()


if predictions is not None and "Date" in predictions.columns:

    prediction_dates = pd.to_datetime(
        predictions["Date"], errors="coerce"
    ).dropna()

    if not prediction_dates.empty:

        latest_prediction_date = prediction_dates.max().date()
        today = pd.Timestamp.now().date()
        age_days = (today - latest_prediction_date).days

        if age_days > 3:
            st.warning(
                f"Prediction data is {age_days} days old "
                f"(latest file date: {latest_prediction_date}). "
                "Refresh predictions before relying on this dashboard."
            )


if predictions is None:

    st.error(
        "Prediction file was not found."
    )

    st.info(
        "Run: python src/14_predict_all_stocks.py"
    )

    st.write(
        "Expected file:"
    )

    st.code(
        "models/all_stock_predictions.csv"
    )

    st.stop()


# ============================================================
# CHECK TICKER COLUMN
# ============================================================

ticker_column = find_column(
    predictions,
    [
        "Ticker",
        "ticker",
        "Symbol",
        "symbol"
    ]
)


if ticker_column is None:

    st.error(
        "Ticker column was not found "
        "in all_stock_predictions.csv."
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(predictions.columns)
    )

    st.stop()


# ============================================================
# NORMALIZE TICKERS
# ============================================================

predictions[ticker_column] = (
    predictions[ticker_column]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# IDENTIFY IMPORTANT COLUMNS
# ============================================================

signal_column = find_column(
    predictions,
    [
        "Signal",
        "signal",
        "Prediction",
        "prediction"
    ]
)

up_column = find_column(
    predictions,
    [
        "UP_%",
        "UP%",
        "Probability_UP",
        "Probability UP",
        "UP_Probability"
    ]
)

down_column = find_column(
    predictions,
    [
        "DOWN_%",
        "DOWN%",
        "Probability_DOWN",
        "Probability DOWN",
        "DOWN_Probability"
    ]
)

price_column = find_column(
    predictions,
    [
        "Price",
        "Close",
        "Latest Price"
    ]
)

rsi_column = find_column(
    predictions,
    [
        "RSI",
        "rsi"
    ]
)

macd_column = find_column(
    predictions,
    [
        "MACD",
        "macd"
    ]
)

risk_column = find_column(
    predictions,
    [
        "Risk_Level",
        "Risk Level",
        "Risk"
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Dashboard Controls"
)


# ------------------------------------------------------------
# Refresh
# ------------------------------------------------------------

if st.sidebar.button(
    "🔄 Refresh Dashboard",
    use_container_width=True
):

    st.cache_data.clear()

    st.rerun()


# ------------------------------------------------------------
# LIVE PREDICTIONS
# ------------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 🔴 Live Predictions"
)

st.sidebar.caption(
    "Fetch real-time prices from Yahoo Finance "
    "and run XGBoost models on the latest data."
)


run_live = st.sidebar.button(
    "⚡ Run Live Predictions",
    use_container_width=True
)


# Auto-refresh controls
auto_refresh = st.sidebar.toggle(
    "Auto-refresh",
    value=False
)

refresh_interval = st.sidebar.slider(
    "Refresh interval (minutes)",
    min_value=1,
    max_value=10,
    value=5,
    disabled=not auto_refresh
)


if run_live:

    progress_bar = st.sidebar.progress(
        0,
        text="Starting live predictions..."
    )

    status_text = st.sidebar.empty()

    def _progress(current, total, ticker):

        pct = int(
            current / total * 100
        ) if total > 0 else 0

        progress_bar.progress(
            min(pct, 100),
            text=(
                f"Processing {ticker}... "
                f"({current}/{total})"
            )
        )

    try:

        live_df = run_live_predictions(
            progress_callback=_progress
        )

        progress_bar.progress(
            100,
            text="✅ Live predictions complete!"
        )

        st.session_state[
            "live_predictions"
        ] = live_df

        st.session_state[
            "live_timestamp"
        ] = pd.Timestamp.now()

        time.sleep(1)

        st.rerun()

    except Exception as error:

        progress_bar.empty()

        st.sidebar.error(
            f"Live prediction failed: {error}"
        )


# Auto-refresh trigger
if auto_refresh:

    last_ts = st.session_state.get(
        "live_timestamp"
    )

    if last_ts is not None:

        elapsed = (
            pd.Timestamp.now() - last_ts
        ).total_seconds()

        if elapsed >= refresh_interval * 60:

            st.session_state[
                "_auto_trigger"
            ] = True

            st.rerun()

    # Handle auto-trigger (runs live on rerun)
    if st.session_state.pop(
        "_auto_trigger",
        False
    ):

        progress_bar = st.sidebar.progress(
            0,
            text="Auto-refreshing..."
        )

        def _auto_progress(
            current, total, ticker
        ):

            pct = int(
                current / total * 100
            ) if total > 0 else 0

            progress_bar.progress(
                min(pct, 100),
                text=(
                    f"Processing {ticker}... "
                    f"({current}/{total})"
                )
            )

        try:

            live_df = run_live_predictions(
                progress_callback=_auto_progress
            )

            progress_bar.progress(
                100,
                text="✅ Auto-refresh complete!"
            )

            st.session_state[
                "live_predictions"
            ] = live_df

            st.session_state[
                "live_timestamp"
            ] = pd.Timestamp.now()

        except Exception as error:

            progress_bar.empty()

            st.sidebar.error(
                f"Auto-refresh failed: {error}"
            )


# Replace static predictions with live data
# if available
if "live_predictions" in st.session_state:

    predictions = st.session_state[
        "live_predictions"
    ]

    predictions = clean_column_names(
        predictions
    )

    live_ts = st.session_state.get(
        "live_timestamp"
    )

    if live_ts is not None:

        age_seconds = int(
            (
                pd.Timestamp.now() - live_ts
            ).total_seconds()
        )

        if age_seconds < 60:
            age_str = f"{age_seconds}s ago"
        else:
            age_str = (
                f"{age_seconds // 60}m "
                f"{age_seconds % 60}s ago"
            )

        st.sidebar.success(
            f"🔴 LIVE — updated {age_str}"
        )

    # Re-identify columns from live data
    ticker_column = find_column(
        predictions,
        [
            "Ticker",
            "ticker",
            "Symbol",
            "symbol"
        ]
    )

    signal_column = find_column(
        predictions,
        [
            "Signal",
            "signal",
            "Prediction",
            "prediction"
        ]
    )

    up_column = find_column(
        predictions,
        [
            "UP_%",
            "UP%",
            "Probability_UP",
            "Probability UP",
            "UP_Probability"
        ]
    )

    down_column = find_column(
        predictions,
        [
            "DOWN_%",
            "DOWN%",
            "Probability_DOWN",
            "Probability DOWN",
            "DOWN_Probability"
        ]
    )

    price_column = find_column(
        predictions,
        [
            "Price",
            "Close",
            "Latest Price"
        ]
    )

    rsi_column = find_column(
        predictions,
        ["RSI", "rsi"]
    )

    macd_column = find_column(
        predictions,
        ["MACD", "macd"]
    )

    risk_column = find_column(
        predictions,
        [
            "Risk_Level",
            "Risk Level",
            "Risk"
        ]
    )

    # Normalize tickers
    if ticker_column is not None:

        predictions[ticker_column] = (
            predictions[ticker_column]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    # Recalculate signal counts
    if signal_column is not None:

        signal_values = (
            predictions[signal_column]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        up_count = signal_values.eq("UP").sum()

        down_count = (
            signal_values.eq("DOWN").sum()
        )

        uncertain_count = (
            signal_values.eq("UNCERTAIN").sum()
        )

    tickers = sorted(
        predictions[ticker_column]
        .dropna()
        .unique()
        .tolist()
    )

    total_stocks = len(tickers)

st.sidebar.markdown("---")


# ------------------------------------------------------------
# Stock selection
# ------------------------------------------------------------

tickers = sorted(
    predictions[ticker_column]
    .dropna()
    .unique()
    .tolist()
)


if len(tickers) == 0:

    st.error(
        "No stocks were found."
    )

    st.stop()


selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    tickers
)


# ============================================================
# SUMMARY CALCULATIONS
# ============================================================

total_stocks = len(tickers)


if signal_column is not None:

    signal_values = (
        predictions[signal_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    up_count = (
        signal_values
        .eq("UP")
        .sum()
    )

    down_count = (
        signal_values
        .eq("DOWN")
        .sum()
    )

    uncertain_count = (
        signal_values
        .eq("UNCERTAIN")
        .sum()
    )

else:

    up_count = 0
    down_count = 0
    uncertain_count = 0


# ============================================================
# MARKET SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Market Summary'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Stocks Analyzed",
        total_stocks
    )


with col2:

    st.metric(
        "UP Signals",
        up_count
    )


with col3:

    st.metric(
        "DOWN Signals",
        down_count
    )


with col4:

    st.metric(
        "UNCERTAIN",
        uncertain_count
    )


# ============================================================
# ALL STOCK PREDICTIONS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'All Stock Predictions'
    '</div>',
    unsafe_allow_html=True
)


display_columns = []

preferred_columns = [

    "Ticker",
    "Date",
    "Price",

    "UP_%",
    "DOWN_%",

    "Signal",

    "RSI",
    "MACD",

    "Volatility",
    "Maximum_Drawdown",
    "Sharpe_Ratio",
    "VaR_95",
    "Beta",
    "ATR",

    "Risk_Level",

    "Rank"
]


for column in preferred_columns:

    if column in predictions.columns:

        display_columns.append(
            column
        )


# Add any columns not explicitly listed
for column in predictions.columns:

    if column not in display_columns:

        display_columns.append(
            column
        )


display_df = predictions[
    display_columns
].copy()


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# STOCK RANKING
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Stock Ranking'
    '</div>',
    unsafe_allow_html=True
)


if os.path.exists(
    RANKING_FILE
):

    try:

        ranking = pd.read_csv(
            RANKING_FILE
        )

        ranking = clean_column_names(
            ranking
        )

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:

        st.warning(
            "Could not read stock ranking file."
        )

        st.code(
            str(error)
        )

else:

    # If ranking file doesn't exist,
    # create a basic ranking from UP probability.

    if up_column is not None:

        ranking = predictions.copy()

        try:

            ranking["_UP_NUMERIC"] = pd.to_numeric(
                ranking[up_column],
                errors="coerce"
            )

            ranking = ranking.sort_values(
                "_UP_NUMERIC",
                ascending=False
            )

            ranking.insert(
                0,
                "Rank",
                range(
                    1,
                    len(ranking) + 1
                )
            )

            ranking_columns = []

            for column in [
                "Rank",
                ticker_column,
                price_column,
                up_column,
                down_column,
                signal_column,
                risk_column
            ]:

                if (
                    column is not None
                    and column in ranking.columns
                    and column not in ranking_columns
                ):

                    ranking_columns.append(
                        column
                    )

            ranking = ranking[
                ranking_columns
            ]

            st.dataframe(
                ranking,
                use_container_width=True,
                hide_index=True
            )

        except Exception:

            st.info(
                "Ranking file not available."
            )

    else:

        st.info(
            "Ranking file not available."
        )


# ============================================================
# SELECTED STOCK
# ============================================================

st.markdown(
    '<div class="section-title">'
    f'{selected_ticker} Analysis'
    '</div>',
    unsafe_allow_html=True
)


selected_rows = predictions[
    predictions[ticker_column]
    == selected_ticker
]


if selected_rows.empty:

    st.warning(
        "No prediction data found for "
        f"{selected_ticker}."
    )

    st.stop()


selected = selected_rows.iloc[-1]


# ============================================================
# SELECTED STOCK MAIN METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Price
# ------------------------------------------------------------

with col1:

    if price_column is not None:

        price = selected[
            price_column
        ]

        try:

            st.metric(
                "Price (prediction date)",
                f"${float(price):,.2f}"
            )

        except Exception:

            st.metric(
                "Price (prediction date)",
                str(price)
            )

    else:

        st.metric(
            "Price (prediction date)",
            "N/A"
        )


# ------------------------------------------------------------
# UP probability
# ------------------------------------------------------------

with col2:

    if up_column is not None:

        st.metric(
            "UP Probability",
            format_percentage(
                selected[
                    up_column
                ]
            )
        )

    else:

        st.metric(
            "UP Probability",
            "N/A"
        )


# ------------------------------------------------------------
# DOWN probability
# ------------------------------------------------------------

with col3:

    if down_column is not None:

        st.metric(
            "DOWN Probability",
            format_percentage(
                selected[
                    down_column
                ]
            )
        )

    else:

        st.metric(
            "DOWN Probability",
            "N/A"
        )


# ------------------------------------------------------------
# Signal
# ------------------------------------------------------------

with col4:

    if signal_column is not None:

        st.metric(
            "Prediction",
            str(
                selected[
                    signal_column
                ]
            )
        )

    else:

        st.metric(
            "Prediction",
            "N/A"
        )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Technical Indicators'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    if rsi_column is not None:

        st.metric(
            "RSI",
            format_number(
                selected[
                    rsi_column
                ],
                2
            )
        )

    else:

        st.metric(
            "RSI",
            "N/A"
        )


with col2:

    if macd_column is not None:

        st.metric(
            "MACD",
            format_number(
                selected[
                    macd_column
                ],
                4
            )
        )

    else:

        st.metric(
            "MACD",
            "N/A"
        )


with col3:

    if risk_column is not None:

        st.metric(
            "Risk Level",
            str(
                selected[
                    risk_column
                ]
            )
        )

    else:

        st.metric(
            "Risk Level",
            "N/A"
        )


# ============================================================
# RISK METRICS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Risk Metrics'
    '</div>',
    unsafe_allow_html=True
)


risk_columns = [

    "Volatility",
    "Maximum_Drawdown",
    "Sharpe_Ratio",
    "VaR_95",
    "Beta",
    "ATR"
]


risk_available = [

    column

    for column in risk_columns

    if column in predictions.columns

]


if len(risk_available) == 0:

    st.info(
        "Risk metrics are not present "
        "in the prediction CSV yet."
    )

else:

    risk_col_count = min(
        len(risk_available),
        3
    )

    for start in range(
        0,
        len(risk_available),
        3
    ):

        row_columns = st.columns(
            min(
                3,
                len(
                    risk_available[
                        start:start + 3
                    ]
                )
            )
        )

        for index, column in enumerate(
            risk_available[
                start:start + 3
            ]
        ):

            with row_columns[index]:

                value = selected[
                    column
                ]

                if column in [
                    "Volatility",
                    "Maximum_Drawdown",
                    "VaR_95"
                ]:

                    display_value = (
                        format_percentage(
                            value
                        )
                    )

                elif column == "Sharpe_Ratio":

                    display_value = (
                        format_number(
                            value,
                            2
                        )
                    )

                elif column == "Beta":

                    display_value = (
                        format_number(
                            value,
                            2
                        )
                    )

                elif column == "ATR":

                    display_value = (
                        format_number(
                            value,
                            2
                        )
                    )

                else:

                    display_value = str(
                        value
                    )

                st.metric(
                    column,
                    display_value
                )


# ============================================================
# HISTORICAL DATA
# ============================================================

historical = get_historical_data(
    selected_ticker
)


if historical is None:

    st.warning(
        f"Historical data for "
        f"{selected_ticker} was not found."
    )

else:

    # --------------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------------

    if "Close" in historical.columns:

        st.markdown(
            '<div class="section-title">'
            f'{selected_ticker} Price History'
            '</div>',
            unsafe_allow_html=True
        )

        price_chart = historical[
            [
                "Date",
                "Close"
            ]
        ].dropna()

        if not price_chart.empty:

            price_chart = (
                price_chart
                .set_index("Date")
            )

            st.line_chart(
                price_chart,
                use_container_width=True
            )


    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    moving_average_columns = [

        column

        for column in [
            "Close",
            "SMA_20",
            "SMA_50",
            "SMA_200"
        ]

        if column in historical.columns
    ]


    if len(
        moving_average_columns
    ) > 1:

        st.markdown(
            '<div class="section-title">'
            'Price and Moving Averages'
            '</div>',
            unsafe_allow_html=True
        )

        ma_chart = historical[
            [
                "Date"
            ]
            +
            moving_average_columns
        ].dropna(
            subset=["Date"]
        )

        ma_chart = (
            ma_chart
            .set_index("Date")
        )

        st.line_chart(
            ma_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # RSI CHART
    # --------------------------------------------------------

    if "RSI" in historical.columns:

        st.markdown(
            '<div class="section-title">'
            'RSI'
            '</div>',
            unsafe_allow_html=True
        )

        rsi_chart = historical[
            [
                "Date",
                "RSI"
            ]
        ].dropna()

        if not rsi_chart.empty:

            rsi_chart = (
                rsi_chart
                .set_index("Date")
            )

            st.line_chart(
                rsi_chart,
                use_container_width=True
            )


    # --------------------------------------------------------
    # MACD CHART
    # --------------------------------------------------------

    macd_columns = [

        column

        for column in [
            "MACD",
            "MACD_Signal"
        ]

        if column in historical.columns
    ]


    if len(macd_columns) > 0:

        st.markdown(
            '<div class="section-title">'
            'MACD'
            '</div>',
            unsafe_allow_html=True
        )

        macd_chart = historical[
            [
                "Date"
            ]
            +
            macd_columns
        ].dropna(
            subset=["Date"]
        )

        macd_chart = (
            macd_chart
            .set_index("Date")
        )

        st.line_chart(
            macd_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # VOLUME CHART
    # --------------------------------------------------------

    if "Volume" in historical.columns:

        st.markdown(
            '<div class="section-title">'
            'Trading Volume'
            '</div>',
            unsafe_allow_html=True
        )

        volume_chart = historical[
            [
                "Date",
                "Volume"
            ]
        ].dropna()

        if not volume_chart.empty:

            volume_chart = (
                volume_chart
                .set_index("Date")
            )

            st.bar_chart(
                volume_chart,
                use_container_width=True
            )


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "View Prediction Data"
):

    st.dataframe(
        selected_rows,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "US Stock Market ML Dashboard | "
    "XGBoost + Technical Indicators + "
    "Risk Metrics + Stock Ranking"
)