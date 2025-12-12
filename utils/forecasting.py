# forecasting.py
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib
import warnings

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

FORECAST_DAYS = 180  # Fixed 6-month forecast


# Try to import auto_arima (pmdarima). If not available, fallback to statsmodels ARIMA.
try:
    from pmdarima import auto_arima

    HAS_AUTO_ARIMA = True
except Exception:
    HAS_AUTO_ARIMA = False


# ================== UTILS ==================

def calculate_confidence(rmse, history):
    """
    Confidence based on RMSE vs historical volatility,
    penalized for long forecast horizon.
    """
    volatility = float(np.std(history)) if np.std(history) > 0 else 1.0
    base_confidence = max(0.0, 1 - (rmse / (volatility + 1e-9)))

    # 6-month uncertainty penalty (tunable)
    confidence = base_confidence * 0.75
    # Bound between 0 and 1
    confidence = max(0.0, min(1.0, confidence))
    return round(confidence, 2)


def detect_trend(history, forecast):
    """
    Detect simple trend by comparing last-7 average of forecast vs history.
    """
    history = np.asarray(history)
    forecast = np.asarray(forecast)
    if len(history) < 7:
        hist_avg = np.mean(history)
    else:
        hist_avg = np.mean(history[-7:])

    if len(forecast) < 7:
        fore_avg = np.mean(forecast)
    else:
        fore_avg = np.mean(forecast[-7:])

    delta = fore_avg - hist_avg
    threshold = np.std(history) * 0.3 if np.std(history) > 0 else 0.1

    if delta > threshold:
        return "Upward"
    elif delta < -threshold:
        return "Downward"
    return "Stable"


def calculate_volatility(values):
    return round(float(np.std(values)), 2)


# ================== CHART ==================

def generate_forecast_chart(df, dates, preds):
    """
    dates: list of datetime objects (forecast days)
    preds: list/array of forecast values (same len as dates)
    returns: base64-encoded PNG string
    """
    plt.figure(figsize=(11, 5))

    # Historical series
    plt.plot(
        df.index,
        df["value"],
        label="Historical",
        linewidth=2,
        alpha=0.85
    )

    # Forecast boundary (today)
    plt.axvline(
        df.index[-1],
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="Today"
    )

    # Forecast line
    plt.plot(
        dates,
        preds,
        linestyle="--",
        linewidth=2.2,
        label="Forecast"
    )

    # Confidence interval (simple +/-10%)
    preds_arr = np.array(preds, dtype=float)
    lower = preds_arr * 0.9
    upper = preds_arr * 1.1

    plt.fill_between(
        dates,
        lower,
        upper,
        alpha=0.25,
        label="Confidence Interval"
    )

    plt.title(" Forecast (6 Months)", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Interest Level (0–100)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


# ================== LSTM ==================

def forecast_lstm(df, forecast_days, look_back=30, epochs=50):
    """
    Returns: (preds_array (len forecast_days), rmse_on_train)
    """
    values = df["value"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # Ensure look_back is sensible
    look_back = min(look_back, max(5, len(df) // 2))

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        return None

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.3),
        LSTM(64),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)
    ]

    model.fit(
        X, y,
        epochs=epochs,
        batch_size=16,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )

    # Forecast recursively
    preds_scaled = []
    last_batch = scaled[-look_back:].reshape(1, look_back, 1)
    for _ in range(forecast_days):
        p = model.predict(last_batch, verbose=0)[0, 0]
        preds_scaled.append(p)
        last_batch = np.append(last_batch[:, 1:, :], [[[p]]], axis=1)

    preds = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)).flatten()
    preds = np.clip(preds, 0, 100)

    # Train in-sample predictions for RMSE
    train_preds_scaled = model.predict(X, verbose=0).flatten()
    train_preds = scaler.inverse_transform(
        train_preds_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, train_preds)))

    return np.array(preds), rmse


# ================== ARIMA ==================

def forecast_arima(df, forecast_days):
    """
    Returns: (preds_array (len forecast_days), rmse_on_in_sample)
    Tries to use pmdarima.auto_arima if available; otherwise falls back to statsmodels ARIMA(2,1,2).
    """
    series = df["value"].astype(float)

    try:
        if HAS_AUTO_ARIMA:
            model = auto_arima(
                series,
                seasonal=False,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                max_p=5,
                max_q=5,
            )
            preds = model.predict(n_periods=forecast_days)
            # in-sample predictions
            in_sample = model.predict_in_sample()
            rmse = float(np.sqrt(mean_squared_error(
                series.values[: len(in_sample)], in_sample)))
            preds = np.clip(preds, 0, 100)
            return np.array(preds), rmse
        else:
            # fallback
            model = ARIMA(series, order=(2, 1, 2))
            fit = model.fit()
            preds = fit.forecast(steps=forecast_days)
            in_sample = fit.predict(start=0, end=len(series) - 1)
            rmse = float(np.sqrt(mean_squared_error(series.values, in_sample)))
            preds = np.clip(np.array(preds, dtype=float), 0, 100)
            return np.array(preds), rmse
    except Exception:
        return None


# ================== PROPHET ==================

def forecast_prophet(df, forecast_days):
    """
    Train Prophet on full data (with sensible seasonalities) and return future preds and RMSE on last 20% of data.
    Expects df.index to be datetime and df has column 'value'.
    """
    prophet_df = df.reset_index().rename(
        columns={df.index.name or "index": "date"})
    # Ensure date column name correct
    if "date" not in prophet_df.columns:
        # attempt to find datetime column
        for c in prophet_df.columns:
            if np.issubdtype(prophet_df[c].dtype, np.datetime64):
                prophet_df = prophet_df.rename(columns={c: "date"})
                break

    prophet_df = prophet_df.rename(
        columns={"date": "ds", "value": "y"})[["ds", "y"]]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.9
        )

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        preds = forecast["yhat"].values[-forecast_days:]
        preds = np.clip(preds, 0, 100)

        # RMSE computed on last 20% of original data (if enough points)
        split = int(len(prophet_df) * 0.8)
        if split < len(prophet_df) - 1:
            # produce predictions for the training portion to compare
            in_sample_forecast = forecast.loc[: len(
                prophet_df) - 1, "yhat"].values
            y_true = prophet_df["y"].values
            # Use only the tail part for RMSE (matching earlier logic)
            rmse = float(np.sqrt(mean_squared_error(
                y_true[split:], in_sample_forecast[split:])))
        else:
            # fallback: compute RMSE on whole in-sample
            in_sample_forecast = forecast.loc[: len(
                prophet_df) - 1, "yhat"].values
            y_true = prophet_df["y"].values
            rmse = float(
                np.sqrt(mean_squared_error(y_true, in_sample_forecast)))

        return np.array(preds), rmse
    except Exception:
        return None


# ================== MULTI-MODEL (Weighted Ensemble) ==================

def multi_model_forecast(df):
    """
    Runs all models, evaluates them, builds a weighted ensemble (inverse-RMSE weights),
    and returns forecast (list of dicts), base64 chart, and insight dict.
    """
    # Ensure df index is datetime and named
    if not np.issubdtype(df.index.dtype, np.datetime64):
        # try to convert if there's a column named 'date' or 'ds'
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"]))
        elif "ds" in df.columns:
            df = df.set_index(pd.to_datetime(df["ds"]))
        else:
            raise ValueError(
                "DataFrame must have a datetime index or a 'date'/'ds' column.")
    df = df.sort_index()
    history = df["value"].astype(float).values.tolist()

    results = {}
    model_errors = {}

    # LSTM
    lstm_res = forecast_lstm(df, FORECAST_DAYS, look_back=30, epochs=50)
    if lstm_res:
        preds, rmse = lstm_res
        results["LSTM"] = np.array(preds)
        model_errors["LSTM"] = float(rmse)

    # ARIMA
    arima_res = forecast_arima(df, FORECAST_DAYS)
    if arima_res:
        preds, rmse = arima_res
        results["ARIMA"] = np.array(preds)
        model_errors["ARIMA"] = float(rmse)

    # PROPHET
    prophet_res = forecast_prophet(df, FORECAST_DAYS)
    if prophet_res:
        preds, rmse = prophet_res
        results["PROPHET"] = np.array(preds)
        model_errors["PROPHET"] = float(rmse)

    if not results:
        raise RuntimeError("No forecasting models produced results.")

    # Print model evaluation
    print("\n🔍 Model Evaluation:")
    for name, rmse in model_errors.items():
        conf = calculate_confidence(rmse, history)
        print(f"{name:<8} → RMSE: {rmse:.4f} | Confidence: {conf}")

    # Build weights = inverse RMSE
    eps = 1e-8
    inv_errors = {k: 1.0 / (v + eps) for k, v in model_errors.items()}
    total_inv = sum(inv_errors.values())
    weights = {k: (w / total_inv) for k, w in inv_errors.items()}

    print("\n🔧 Ensemble Weights (inverse-RMSE normalized):")
    for k, w in weights.items():
        print(f"  {k}: {w:.3f}")

    # Ensure all preds arrays have same length
    model_names = list(results.keys())
    # shape (n_models, forecast_days)
    preds_matrix = np.vstack([results[m] for m in model_names])

    # Weighted ensemble
    weight_array = np.array([weights[m] for m in model_names]).reshape(-1, 1)
    ensemble_preds = (preds_matrix * weight_array).sum(axis=0)
    ensemble_preds = np.clip(ensemble_preds, 0, 100)

    # Compute ensemble "rmse" proxy as weighted average of model RMSEs
    ensemble_rmse = float(
        sum(model_errors[m] * weights[m] for m in model_names))

    # Select best single model (lowest rmse)
    best_name = min(model_errors.items(), key=lambda x: x[1])[0]
    best_rmse = model_errors[best_name]

    print(f"\n✅ Selected Best Model: {best_name} (RMSE {best_rmse:.4f})")
    print("✅ Returning weighted ensemble forecast + best-model metadata.\n")

    # Dates for forecast
    last_date = df.index[-1]
    dates = [last_date + timedelta(days=i + 1) for i in range(FORECAST_DAYS)]

    # Prepare forecast list using ensemble predictions
    forecast_list = [
        {"date": d.strftime("%Y-%m-%d"), "predicted_value": round(float(p), 2)}
        for d, p in zip(dates, ensemble_preds)
    ]

    insight = {
        "model": "ENSEMBLE",
        "ensemble_components": {m: {"weight": round(float(weights[m]), 4), "rmse": round(float(model_errors[m]), 4)} for m in model_names},
        "best_single_model": best_name,
        "best_single_model_rmse": round(float(best_rmse), 4),
        "ensemble_rmse_proxy": round(float(ensemble_rmse), 4),
        "confidence": calculate_confidence(ensemble_rmse, history),
        "trend": detect_trend(history, ensemble_preds),
        "volatility": calculate_volatility(ensemble_preds),
    }

    chart = generate_forecast_chart(df, dates, ensemble_preds)

    return {
        "forecast": forecast_list,
        "chart": chart,
        "insight": insight
    }
