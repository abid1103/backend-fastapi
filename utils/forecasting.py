# forecasting.py

from datetime import timedelta
import base64
import io
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from prophet import Prophet

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ================= CONFIG =================
FORECAST_WEEKS = 26          # 6 months (Google Trends is WEEKLY)
MAX_HISTORY_POINTS = 200     # Speed + stability

# ================= UTILS =================


def calculate_confidence(rmse, history):
    volatility = float(np.std(history)) if np.std(history) > 0 else 1.0
    score = 1 - (rmse / (volatility + 1e-9))
    return round(max(0.0, min(score, 1.0)), 2)


def detect_trend(history, forecast):
    hist_avg = np.mean(history[-4:])
    fore_avg = np.mean(forecast[:4])
    delta = fore_avg - hist_avg
    threshold = np.std(history) * 0.2

    if delta > threshold:
        return "Upward"
    elif delta < -threshold:
        return "Downward"
    return "Stable"


def calculate_volatility(values):
    return round(float(np.std(values)), 2)

# ================= CHART =================


def generate_chart(df, dates, preds):
    plt.figure(figsize=(12, 5))

    # Historical
    plt.plot(df.index, df["value"], label="Historical", linewidth=2)

    # Today marker
    plt.axvline(df.index[-1], linestyle="--", alpha=0.6, label="Today")

    # Forecast
    plt.plot(dates, preds, linestyle="--", linewidth=2, label="Forecast")

    # Honest uncertainty band (±1 std)
    std = np.std(preds)
    lower = np.clip(preds - std, 0, 100)
    upper = np.clip(preds + std, 0, 100)
    plt.fill_between(dates, lower, upper, alpha=0.2, label="Uncertainty Range")

    plt.title("Google Trends Forecast")
    plt.xlabel("Date")
    plt.ylabel("Interest")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ================= MODEL =================


def forecast_prophet(df, weeks):
    pdf = df.reset_index().rename(
        columns={df.index.name or "date": "ds", "value": "y"}
    )
    pdf["ds"] = pd.to_datetime(pdf["ds"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        interval_width=0.9
    )

    model.fit(pdf)

    future = model.make_future_dataframe(periods=weeks, freq="W")
    forecast = model.predict(future)

    preds = forecast["yhat"].values[-weeks:]
    preds = np.clip(preds, 0, 100)

    # RMSE on last 20% of history
    split = int(len(pdf) * 0.8)
    rmse = np.sqrt(mean_squared_error(
        pdf["y"].values[split:],
        forecast["yhat"].values[split:len(pdf)]
    ))

    return preds, float(rmse)

# ================= MAIN ENTRY =================


def multi_model_forecast(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().tail(MAX_HISTORY_POINTS)

    history = df["value"].astype(float).values

    # Prophet forecast
    preds, rmse = forecast_prophet(df, FORECAST_WEEKS)

    last_date = df.index[-1]
    dates = [last_date + timedelta(weeks=i + 1) for i in range(FORECAST_WEEKS)]

    forecast = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "predicted_value": round(float(v), 2)
        }
        for d, v in zip(dates, preds)
    ]

    insight = {
        "model": "PROPHET",
        "best_model": "PROPHET",
        "confidence": calculate_confidence(rmse, history),
        "trend": detect_trend(history, preds),
        "volatility": calculate_volatility(preds),
    }

    chart = generate_chart(df, dates, preds)

    return {
        "forecast": forecast,
        "chart": chart,
        "insight": insight
    }
