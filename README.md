# 🚀 Future Forecast Pro: Enterprise-Grade Sales Intelligence

![Project Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![Tech Stack](https://img.shields.io/badge/Stack-MLOps--Forecasting-blue)
![API](https://img.shields.io/badge/API-FastAPI-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

An end-to-end, industrial-strength time-series forecasting platform. This project goes beyond simple predictions, offering a full MLOps pipeline from automated feature engineering to RESTful API deployment.

---

## 🌟 Key Features

### 🧠 Advanced Model Engine
- **Bayesian Hyperparameter Tuning**: Uses **Optuna** to automatically optimize Prophet parameters (`changepoint_prior_scale`, etc.).
- **Model Benchmarking**: Automatically compares **Prophet** against **XGBoost** and selects the best performer based on RMSE/MAPE.
- **Probabilistic Forecasting**: Provides uncertainty intervals for risk-aware decision making.
- **Cross-Validation**: Implements rolling-window validation to ensure model robustness.

### 🏗️ Intelligent Data Pipeline
- **Automated Feature Engineering**: Generates lag features, rolling averages, and seasonal indicators.
- **Data Validation & Cleaning**: Automatic detection of missing values, duplicates, and inconsistent timestamps.
- **Outlier Treatment**: Uses **Isolation Forests** and Winsorization to prevent model drift from "garbage" data.
- **Synthetic Data Engine**: Advanced generator with controllable trends, noise, and seasonal shifts.

### 📊 Visualization & BI
- **Interactive Dashboard**: Built with **Streamlit** and **Plotly** for dynamic data exploration.
- **Decomposition Analysis**: Visualize trend, seasonality, and residuals separately.
- **Business Intelligence Layer**: Automated inventory recommendations and demand drop alerts.
- **Scenario Simulation**: Test the impact of different trends and noise levels in real-time.

### 🚢 Deployment & Usability
- **REST API**: Production-ready **FastAPI** service for serving predictions on-demand.
- **CLI Interface**: Powerful command-line tool for data generation, serving, and dashboard launching.
- **Containerization**: Full **Docker** support for seamless deployment across environments.
- **Logging & Monitoring**: Retraining triggers and health checks built-in.

---

## 🏗️ Project Structure

```
FUTURE_ML_01/
├── src/
│   ├── data/            # Ingestion, validation, and feature engineering
│   ├── models/          # Model engine, tuning, and benchmarking
│   ├── viz/             # Streamlit dashboard and Plotly interactive plots
│   ├── api/             # FastAPI REST service implementation
│   └── utils/           # Shared metrics and utility functions
├── main.py              # CLI entry point
├── Dockerfile           # Production container configuration
├── requirements.txt     # Enterprise-grade dependency list
└── FUTURE_ML_01.ipynb   # Interactive analysis notebook
```

---

## 🛠️ Tech Stack

- **ML Frameworks**: Prophet, XGBoost, Scikit-Learn, Statsmodels
- **Optimization**: Optuna (Bayesian Tuning)
- **API**: FastAPI, Uvicorn
- **UI/Dash**: Streamlit, Plotly
- **Infrastructure**: Docker, Typer (CLI)

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard (Recommended)
```bash
python main.py dashboard
```

### 3. Start the API Server
```bash
python main.py serve --port 8000
```

### 4. Generate Synthetic Data
```bash
python main.py generate --periods 500 --output my_data.csv
```

---

## 📈 Performance Benchmarks

In our latest production simulation, the engine benchmarked multiple architectures on a 350-day dataset:

| Model | Tuning Strategy | RMSE | Status |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Lag Feature Engineering | **17.48** | 🏆 **Production** |
| **Prophet** | Bayesian Optuna Tuning | 26.78 | Candidate |

### 🔍 Top Predictive Factors
The system's Explainable AI (XAI) layer identified that **7-day rolling trends** and **weekly seasonality** account for over **83%** of the forecast accuracy.

---
Created with ❤️ by [Santanu](https://github.com/santanu949)
