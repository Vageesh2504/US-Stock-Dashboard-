"""
Live prediction module.

Downloads fresh market data via yfinance, computes technical features,
loads the trained XGBoost models, and returns a DataFrame of live
predictions with risk metrics — all callable from the Streamlit dashboard.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

_THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

_BASE_DIR = os.path.dirname(
    _THIS_DIR
)

MODEL_FOLDER = os.path.join(
    _BASE_DIR,
    "models"
)


# ============================================================
# STOCKS & FEATURES
# ============================================================

STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "V",
    "WMT"
]

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",

    "SMA_5",
    "SMA_20",
    "SMA_50",
    "SMA_200",

    "EMA_12",
    "EMA_26",

    "RSI",

    "MACD",
    "MACD_Signal",
    "MACD_Difference",

    "ATR",

    "BB_High",
    "BB_Low",
    "BB_Middle",

    "Volatility_20D",

    "Volume_Change",
    "Volume_Ratio",

    "Distance_SMA20",
    "Distance_SMA50",
    "Distance_SMA200",

    "SPY_Return",
    "SPY_Volatility",

    "QQQ_Return",
    "QQQ_Volatility",

    "DIA_Return",
    "DIA_Volatility",

    "VIX_Return",
    "VIX_Volatility"
]

HISTORY_PERIOD = "5y"


# ============================================================
# TECHNICAL HELPERS
# ============================================================

def _calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(period).mean()
    average_loss = loss.rolling(period).mean()

    rs = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    return 100 - (100 / (1 + rs))


def _calculate_atr(df, period=14):

    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - previous_close).abs()
    tr3 = (df["Low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def _download_daily(ticker):

    data = yf.download(
        ticker,
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No historical data for {ticker}"
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = (
            data.columns.get_level_values(0)
        )

    data = data.reset_index()

    if "Date" not in data.columns:
        data = data.rename(
            columns={data.columns[0]: "Date"}
        )

    data["Date"] = pd.to_datetime(data["Date"])

    try:
        if data["Date"].dt.tz is not None:
            data["Date"] = (
                data["Date"].dt.tz_localize(None)
            )
    except Exception:
        pass

    return data


def _get_realtime(ticker):

    data = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No real-time data for {ticker}"
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = (
            data.columns.get_level_values(0)
        )

    data = data.dropna(subset=["Close"])

    if data.empty:
        raise ValueError(
            f"Real-time data has no valid prices "
            f"for {ticker}"
        )

    latest = data.iloc[-1]

    return {
        "Price": float(latest["Close"]),
        "Open": float(latest["Open"]),
        "High": float(latest["High"]),
        "Low": float(latest["Low"]),
        "Volume": float(latest["Volume"])
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _create_features(stock_df, spy, qqq, dia, vix):

    df = stock_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]

    # Returns
    df["Return_1D"] = close.pct_change(1)
    df["Return_5D"] = close.pct_change(5)
    df["Return_10D"] = close.pct_change(10)
    df["Return_20D"] = close.pct_change(20)

    # SMA
    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    # EMA
    df["EMA_12"] = close.ewm(
        span=12, adjust=False
    ).mean()

    df["EMA_26"] = close.ewm(
        span=26, adjust=False
    ).mean()

    # RSI
    df["RSI"] = _calculate_rsi(close, 14)

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_Difference"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # ATR
    df["ATR"] = _calculate_atr(df, 14)

    # Bollinger Bands
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()

    df["BB_Middle"] = rolling_mean
    df["BB_High"] = rolling_mean + 2 * rolling_std
    df["BB_Low"] = rolling_mean - 2 * rolling_std

    # Volatility
    df["Volatility_20D"] = (
        df["Return_1D"].rolling(20).std()
    )

    # Volume
    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    volume_avg = df["Volume"].rolling(20).mean()

    df["Volume_Ratio"] = (
        df["Volume"] / volume_avg
    )

    # Distance from MAs
    df["Distance_SMA20"] = close / df["SMA_20"] - 1
    df["Distance_SMA50"] = close / df["SMA_50"] - 1
    df["Distance_SMA200"] = close / df["SMA_200"] - 1

    # Market features
    market_data = {
        "SPY": spy,
        "QQQ": qqq,
        "DIA": dia,
        "VIX": vix
    }

    for name, market_df in market_data.items():

        market = market_df[["Date", "Close"]].copy()
        market = market.sort_values("Date")

        market["Return"] = (
            market["Close"].pct_change()
        )

        market["Volatility"] = (
            market["Return"].rolling(20).std()
        )

        df = df.merge(
            market[
                ["Date", "Return", "Volatility"]
            ].rename(columns={
                "Return": f"{name}_Return",
                "Volatility": f"{name}_Volatility"
            }),
            on="Date",
            how="left"
        )

    return df


def _update_with_realtime(features_df, realtime):

    df = features_df.copy()
    idx = df.index[-1]

    df.loc[idx, "Open"] = realtime["Open"]
    df.loc[idx, "High"] = realtime["High"]
    df.loc[idx, "Low"] = realtime["Low"]
    df.loc[idx, "Close"] = realtime["Price"]
    df.loc[idx, "Volume"] = realtime["Volume"]

    close = df["Close"]

    df["Return_1D"] = close.pct_change()
    df["Return_5D"] = close.pct_change(5)
    df["Return_10D"] = close.pct_change(10)
    df["Return_20D"] = close.pct_change(20)

    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    df["EMA_12"] = close.ewm(
        span=12, adjust=False
    ).mean()

    df["EMA_26"] = close.ewm(
        span=26, adjust=False
    ).mean()

    df["RSI"] = _calculate_rsi(close, 14)

    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_Difference"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    df["ATR"] = _calculate_atr(df, 14)

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()

    df["BB_Middle"] = rolling_mean
    df["BB_High"] = rolling_mean + 2 * rolling_std
    df["BB_Low"] = rolling_mean - 2 * rolling_std

    df["Volatility_20D"] = (
        df["Return_1D"].rolling(20).std()
    )

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        / df["Volume"].rolling(20).mean()
    )

    df["Distance_SMA20"] = close / df["SMA_20"] - 1
    df["Distance_SMA50"] = close / df["SMA_50"] - 1
    df["Distance_SMA200"] = close / df["SMA_200"] - 1

    return df


# ============================================================
# RISK METRICS (inlined to avoid import path issues)
# ============================================================

def _risk_metrics(stock_df, market_df):

    df = stock_df.copy().sort_values("Date")
    returns = df["Close"].pct_change().dropna()

    if len(returns) < 30:
        return {
            "Volatility": np.nan,
            "Maximum_Drawdown": np.nan,
            "Sharpe_Ratio": np.nan,
            "VaR_95": np.nan,
            "Beta": np.nan,
            "ATR": np.nan
        }

    daily_vol = returns.std()
    ann_vol = daily_vol * np.sqrt(252)

    cum_max = df["Close"].cummax()
    drawdown = df["Close"] / cum_max - 1
    max_dd = drawdown.min()

    ann_ret = returns.mean() * 252
    sharpe = (
        ann_ret / ann_vol
        if ann_vol != 0
        else np.nan
    )

    var_95 = returns.quantile(0.05)

    # Beta
    beta = np.nan

    if market_df is not None:

        stock_ret = (
            df.set_index("Date")["Close"]
            .pct_change()
            .rename("S")
        )

        mkt = market_df.copy().sort_values("Date")
        mkt_ret = (
            mkt.set_index("Date")["Close"]
            .pct_change()
            .rename("M")
        )

        combined = pd.concat(
            [stock_ret, mkt_ret], axis=1
        ).dropna()

        if len(combined) >= 30:
            mkt_var = combined["M"].var()
            if mkt_var != 0:
                beta = (
                    combined["S"].cov(combined["M"])
                    / mkt_var
                )

    # ATR (last value)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_val = tr.rolling(14).mean().iloc[-1]

    return {
        "Volatility": ann_vol,
        "Maximum_Drawdown": max_dd,
        "Sharpe_Ratio": sharpe,
        "VaR_95": var_95,
        "Beta": beta,
        "ATR": atr_val
    }


def _risk_level(volatility, max_dd, var_95):

    v = abs(volatility) if pd.notna(volatility) else 0
    d = abs(max_dd) if pd.notna(max_dd) else 0
    r = abs(var_95) if pd.notna(var_95) else 0

    if v >= 0.40 or d >= 0.30 or r >= 0.04:
        return "HIGH"

    if v >= 0.20 or d >= 0.15 or r >= 0.02:
        return "MEDIUM"

    return "LOW"


# ============================================================
# PREDICT ONE STOCK
# ============================================================

def _predict_one(ticker, spy, qqq, dia, vix):

    # Load model
    model_path = os.path.join(
        MODEL_FOLDER,
        f"{ticker}_xgboost.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(model_path)

    # Data
    stock_data = _download_daily(ticker)
    realtime = _get_realtime(ticker)

    features_df = _create_features(
        stock_data, spy, qqq, dia, vix
    )

    features_df = _update_with_realtime(
        features_df, realtime
    )

    # Check features
    missing = [
        f for f in FEATURES
        if f not in features_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing features: "
            + ", ".join(missing)
        )

    model_input = (
        features_df[FEATURES]
        .iloc[-1:]
        .copy()
    )

    if model_input.isna().any().any():
        nan_cols = (
            model_input.columns[
                model_input.isna().any()
            ].tolist()
        )
        raise ValueError(
            "NaN in features: "
            + ", ".join(nan_cols)
        )

    latest = features_df.iloc[-1]

    # Predict
    probabilities = model.predict_proba(
        model_input
    )[0]

    classes = list(model.classes_)

    if 1 in classes:
        up_prob = probabilities[classes.index(1)]
    else:
        up_prob = float(probabilities[-1])

    down_prob = 1 - up_prob

    if up_prob >= 0.60:
        signal = "UP"
    elif up_prob <= 0.40:
        signal = "DOWN"
    else:
        signal = "UNCERTAIN"

    # Risk
    market_close = spy[["Date", "Close"]].copy()
    market_close["Date"] = pd.to_datetime(
        market_close["Date"]
    )

    risk = _risk_metrics(stock_data, market_close)

    rl = _risk_level(
        risk["Volatility"],
        risk["Maximum_Drawdown"],
        risk["VaR_95"]
    )

    return {
        "Ticker": ticker,
        "Date": str(pd.Timestamp.now().date()),
        "Price": float(realtime["Price"]),
        "UP_%": float(up_prob * 100),
        "DOWN_%": float(down_prob * 100),
        "Signal": signal,
        "RSI": float(latest["RSI"]),
        "MACD": float(latest["MACD"]),
        "Volatility": float(risk["Volatility"]),
        "Maximum_Drawdown": float(
            risk["Maximum_Drawdown"]
        ),
        "Sharpe_Ratio": float(
            risk["Sharpe_Ratio"]
        ),
        "VaR_95": float(risk["VaR_95"]),
        "Beta": float(risk["Beta"]),
        "ATR": float(risk["ATR"]),
        "Risk_Level": rl
    }


# ============================================================
# PUBLIC API — run_live_predictions()
# ============================================================

def run_live_predictions(
    progress_callback=None
):
    """
    Download live data, run XGBoost models, and return
    a DataFrame with fresh predictions + risk metrics.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with (current_index, total, ticker_name)
        so the UI can display a progress bar.

    Returns
    -------
    pd.DataFrame
        One row per stock with columns matching the
        static all_stock_predictions.csv format.
    """

    # Market indicators
    spy = _download_daily("SPY")
    qqq = _download_daily("QQQ")
    dia = _download_daily("DIA")
    vix = _download_daily("^VIX")

    results = []
    total = len(STOCKS)

    for idx, ticker in enumerate(STOCKS):

        if progress_callback is not None:
            progress_callback(idx, total, ticker)

        try:

            result = _predict_one(
                ticker, spy, qqq, dia, vix
            )

            results.append(result)

        except Exception as exc:

            results.append({
                "Ticker": ticker,
                "Date": str(
                    pd.Timestamp.now().date()
                ),
                "Price": np.nan,
                "UP_%": np.nan,
                "DOWN_%": np.nan,
                "Signal": "ERROR",
                "RSI": np.nan,
                "MACD": np.nan,
                "Volatility": np.nan,
                "Maximum_Drawdown": np.nan,
                "Sharpe_Ratio": np.nan,
                "VaR_95": np.nan,
                "Beta": np.nan,
                "ATR": np.nan,
                "Risk_Level": f"ERROR: {exc}"
            })

    if progress_callback is not None:
        progress_callback(total, total, "Done")

    return pd.DataFrame(results)
