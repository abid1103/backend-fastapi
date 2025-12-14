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
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ================= CONFIG =================

FORECAST_DAYS = 60          # API-safe horizon
MAX_HISTORY_POINTS = 120    # limit data size for speed

# ================= UTILS =================


def calculate_confidence(rmse, history):
    volatility = float(np.std(history)) if np.std(history) > 0 else 1.0
    base = max(0.0, 1 - (rmse / (volatility + 1e-9)))
    return round(min(1.0, base * 0.8), 2)


def detect_trend(history, forecast):
    hist_avg = np.mean(history[-7:]) if len(history) >= 7 else np.mean(history)
    fore_avg = np.mean(
        forecast[-7:]) if len(forecast) >= 7 else np.mean(forecast)

    delta = fore_avg - hist_avg
    threshold = np.std(history) * 0.25 if np.std(history) > 0 else 0.1

    if delta > threshold:
        return "Upward"
    if delta < -threshold:
        return "Downward"
    return "Stable"


def calculate_volatility(values):
    return round(float(np.std(values)), 2)

# ================= CHART =================


def generate_chart(df, dates, preds):
    plt.figure(figsize=(10, 5))

    plt.plot(df.index, df["value"], label="Historical", linewidth=2)
    plt.axvline(df.index[-1], linestyle="--", alpha=0.6)
    plt.plot(dates, preds, linestyle="--", label="Forecast")

    plt.title("Google Trends Forecast (60 days)")
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

# ================= MODELS =================


def forecast_arima(series, days):
    try:
        model = ARIMA(series, order=(1, 1, 1))
        fit = model.fit()

        preds = fit.forecast(steps=days)
        in_sample = fit.predict(start=0, end=len(series) - 1)

        rmse = float(np.sqrt(mean_squared_error(series, in_sample)))
        preds = np.clip(preds, 0, 100)

        return np.array(preds), rmse
    except Exception:
        return None


def forecast_prophet(df, days):
    try:
        pdf = df.reset_index().rename(columns={"date": "ds", "value": "y"})
        pdf["ds"] = pd.to_datetime(pdf["ds"])

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1
        )

        model.fit(pdf)

        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        preds = forecast["yhat"].values[-days:]
        preds = np.clip(preds, 0, 100)

        split = int(len(pdf) * 0.8)
        rmse = float(np.sqrt(mean_squared_error(
            pdf["y"].values[split:],
            forecast["yhat"].values[split:len(pdf)]
        )))

        return np.array(preds), rmse
    except Exception:
        return None

# ================= ENSEMBLE =================


def multi_model_forecast(df):
    # --- sanitize dataframe ---
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().tail(MAX_HISTORY_POINTS)

    history = df["value"].astype(float).values.tolist()

    results = {}
    errors = {}

    # Prophet
    res = forecast_prophet(df, FORECAST_DAYS)
    if res:
        preds, rmse = res
        results["PROPHET"] = preds
        errors["PROPHET"] = rmse

    # ARIMA
    res = forecast_arima(df["value"].astype(float).values, FORECAST_DAYS)
    if res:
        preds, rmse = res
        results["ARIMA"] = preds
        errors["ARIMA"] = rmse

    if not results:
        raise RuntimeError("Forecasting failed")

    # --- weighted ensemble ---
    inv = {k: 1 / (v + 1e-8) for k, v in errors.items()}
    total = sum(inv.values())
    weights = {k: inv[k] / total for k in inv}

    preds_matrix = np.vstack([results[k] for k in weights])
    weight_arr = np.array([weights[k] for k in weights]).reshape(-1, 1)

    ensemble = (preds_matrix * weight_arr).sum(axis=0)
    ensemble = np.clip(ensemble, 0, 100)

    # --- metadata ---
    best_model = min(errors.items(), key=lambda x: x[1])[0]
    ensemble_rmse = sum(errors[k] * weights[k] for k in weights)

    last_date = df.index[-1]
    dates = [last_date + timedelta(days=i + 1) for i in range(FORECAST_DAYS)]

    forecast = [
        {"date": d.strftime("%Y-%m-%d"), "predicted_value": round(float(v), 2)}
        for d, v in zip(dates, ensemble)
    ]

    insight = {
        "model": "ENSEMBLE",
        "best_model": best_model,
        "confidence": calculate_confidence(ensemble_rmse, history),
        "trend": detect_trend(history, ensemble),
        "volatility": calculate_volatility(ensemble),
    }

    chart = generate_chart(df, dates, ensemble)

    return {
        "forecast": forecast,
        "chart": chart,
        "insight": insight
    }
