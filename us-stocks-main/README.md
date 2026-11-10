# US Stock Market ML Dashboard

A Python project that downloads historical US stock data, engineers technical indicators, trains XGBoost classifiers to estimate the next trading day's direction, computes risk metrics, ranks stocks, and displays results in a Streamlit dashboard.

> **Disclaimer:** This is an educational machine-learning project, not financial advice. Predictions are uncertain, historical performance does not guarantee future results, and the included output CSVs/models may be stale. Do your own research before making financial decisions.

## Features

- Historical market data from Yahoo Finance (`yfinance`) for AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, V, WMT, and market benchmarks.
- Data cleaning and technical features (returns, moving averages, RSI, MACD, ATR, Bollinger Bands, volatility, and volume ratios).
- XGBoost model training with a chronological 80/20 train/test split.
- Prediction probabilities and risk statistics, including volatility, maximum drawdown, Sharpe ratio, historical VaR, beta, and ATR.
- Streamlit dashboard for reviewing stock predictions and rankings.

## Project structure

```text
data/
  raw/                    # Downloaded historical data
  processed/              # Cleaned data and engineered features
models/                   # Trained models and prediction/ranking CSVs
outputs/                  # Backtest outputs and charts
src/                      # Data pipeline, model, risk, ranking, and dashboard code
requirements.txt
```

## Setup (Windows / macOS / Linux)

Use Python 3.10 or 3.11 for broad compatibility with the scientific Python packages. From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard reads `models/all_stock_predictions.csv`, processed historical data under `data/processed/final/`, and optionally `models/stock_rankings.csv`. The repository contains sample/generated outputs; refresh them by running the pipeline below.

## Rebuild data and models

Run these commands from the project root, in order:

```bash
python src/01_download_data.py
python src/02_check_data.py
python src/03_clean_data.py
python src/04_feature_engineering.py
python src/05_add_market_features.py
python src/06_create_dataset.py
python src/13_train_all_stocks.py
python src/14_predict_all_stocks.py
python src/stock_ranking.py
```

`yfinance` requires an internet connection and can return delayed, missing, or rate-limited data. The real-time data helper is `src/realtime_data.py`.

## Deployment on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open Streamlit Community Cloud and create an app from that repository.
3. Set the main file path to `src/dashboard.py`.
4. Ensure `requirements.txt`, the `models/` prediction CSV, and the historical CSV files used by the dashboard are committed.
5. Deploy. For a smaller repository, consider storing large generated data/model artifacts outside Git and downloading them during app startup.

The dashboard is intended to display the latest prediction file available in the repository; it does not automatically retrain models on each page load.
