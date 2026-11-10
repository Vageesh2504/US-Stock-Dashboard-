# 📈 US Stock Market ML Dashboard

🚀 **Live Demo:** https://us-stock-dashboard.streamlit.app

A Python-based machine learning project that downloads historical US stock market data, engineers technical indicators, trains XGBoost classifiers to estimate the next trading day's direction, calculates financial risk metrics, ranks stocks, and displays results through an interactive Streamlit dashboard.

> **Disclaimer:** This is an educational machine-learning project, not financial advice. Predictions are uncertain, historical performance does not guarantee future results, and the included output CSVs/models may be stale.

## ✨ Features

- Historical market data from Yahoo Finance (`yfinance`) for AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, V, WMT, and market benchmarks.
- Data cleaning and technical indicators including RSI, MACD, ATR, Bollinger Bands, moving averages, volatility, and volume ratios.
- XGBoost model training with chronological 80/20 train/test splitting.
- Stock prediction probabilities and UP/DOWN signals.
- Risk analysis including volatility, maximum drawdown, Sharpe ratio, historical VaR, beta, and ATR.
- Stock ranking based on prediction results.
- Interactive Streamlit dashboard with historical charts and stock analysis.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Interactive dashboard |
| XGBoost | Machine learning predictions |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| Yahoo Finance (yfinance) | Stock market data |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| TA | Technical indicators |
| Joblib | Model serialization |

## 📂 Project Structure

```text
US-Stock-Dashboard/
│
├── us-stocks-main/
│   ├── .streamlit/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── models/
│   ├── outputs/
│   ├── src/
│   │   ├── dashboard.py
│   │   ├── live_predict.py
│   │   ├── realtime_data.py
│   │   ├── risk_metrics.py
│   │   └── stock_ranking.py
│   ├── README.md
│   └── requirements.txt
│
└── requirements.txt
```

## ⚙️ Installation and Setup

Use Python 3.10 or 3.11 for compatibility with scientific Python packages.

Clone the repository:

```bash
git clone https://github.com/Vageesh2504/US-Stock-Dashboard-.git
```

Navigate into the project:

```bash
cd US-Stock-Dashboard-/us-stocks-main
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Run the Dashboard

From the `us-stocks-main` directory:

```bash
streamlit run src/dashboard.py
```

The dashboard reads prediction results from `models/all_stock_predictions.csv`, historical data from `data/processed/final/`, and optionally stock rankings from `models/stock_rankings.csv`.

## 🧠 Rebuild Data and Train Models

Run these commands from the `us-stocks-main` directory in order:

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

Yahoo Finance requires an internet connection and may return delayed, missing, or rate-limited data.

## ☁️ Deployment – Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select the `main` branch.
5. Set the main file path to:

```text
us-stocks-main/src/dashboard.py
```

6. Ensure the root `requirements.txt`, prediction CSV files, trained models, and historical datasets are committed.
7. Click Deploy.

## 📊 Project Information

- **Project:** US Stock Market ML Dashboard
- **Domain:** Artificial Intelligence and Machine Learning
- **Category:** Financial Data Analytics
- **Machine Learning Algorithm:** XGBoost
- **Deployment:** Streamlit Community Cloud
- **Repository:** https://github.com/Vageesh2504/US-Stock-Dashboard-

The dashboard displays the latest available prediction files. It does not automatically retrain models on every page load.
