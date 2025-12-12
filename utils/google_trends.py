# utils/google_trends.py

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------------
# In-memory cache (14 days)
# -----------------------------------
TRENDS_CACHE = {}

# -----------------------------------
# API Key
# -----------------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
if not SERPAPI_KEY:
    raise RuntimeError("SERPAPI_KEY not set in environment variables.")


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(df: pd.DataFrame, column: str):
    """Simple hybrid anomaly detection: Z-score + rolling mean spikes."""
    anomalies = []
    if df.empty:
        return anomalies

    values = df[column]
    mean = values.mean()
    std = values.std() if values.std() != 0 else 1
    z = (values - mean) / std
    rolling = values.rolling(window=7, min_periods=3).mean()

    for i, (_, row) in enumerate(df.iterrows()):
        spike = False

        # strong z score outlier
        if abs(z.iloc[i]) > 2.5:
            spike = True

        # rolling mean anomaly
        if not spike and i > 0 and not pd.isna(rolling.iloc[i]):
            if row[column] > rolling.iloc[i] * 1.4:
                spike = True

        if spike:
            anomalies.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "type": "spike"
            })

    return anomalies


# ============================================================
# FETCH GOOGLE TRENDS (SerpAPI)
# ============================================================

def fetch_google_trends(keyword: str, geo: str = ""):
    """
    Fetches Google Trends through SerpAPI.
    Supports caching + DB integration inside main.py.
    """

    cache_key = (keyword.lower(), geo.upper())

    # ----------------------------------------------
    # RETURN CACHED (if fresh)
    # ----------------------------------------------
    if cache_key in TRENDS_CACHE:
        c = TRENDS_CACHE[cache_key]
        if (datetime.utcnow() - c["fetched_at"]).days <= 14:
            return c["trends"], c["anomalies"]

    print(f"Fetching Google Trends via SerpAPI for {keyword} (geo={geo})")
    time.sleep(2)  # human-like delay

    # Build date range: 2 years
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=730)
    date_range = f"{start_date} {end_date}"

    params = {
        "engine": "google_trends",
        "q": keyword,
        "hl": "en",
        "tz": "360",
        "geo": geo.upper(),
        "date": date_range,
        "api_key": SERPAPI_KEY
    }

    try:
        response = requests.get(
            "https://serpapi.com/search", params=params, timeout=30)
        data = response.json()
        print("SerpAPI Raw Data:", data)

        # ----------------------------------------------------
        # Handle SerpAPI errors
        # ----------------------------------------------------
        if "error" in data:
            print("SerpAPI error:", data["error"])
            return [], []

        # ----------------------------------------------------
        # Correct structure from SerpAPI
        # ----------------------------------------------------
        iot = data.get("interest_over_time", {})
        timeline = iot.get("timeline_data", [])

        if not timeline:
            print("SerpAPI returned no timeline_data")
            return [], []

        # ----------------------------------------------------
        # Transform into uniform structure
        # ----------------------------------------------------
        parsed = []
        for item in timeline:
            ts = int(item["timestamp"])
            dt = datetime.utcfromtimestamp(ts)

            # all items have [{"query", "value", "extracted_value"}]
            extracted = item["values"][0].get("extracted_value")
            if extracted is None:
                continue

            parsed.append({"date": dt, "value": int(extracted)})

        if not parsed:
            print("No usable trend values after parsing.")
            return [], []

        df = pd.DataFrame(parsed).sort_values("date")

        # ----------------------------------------------------
        # Resample to ~2–3 points per week (smooth)
        # ----------------------------------------------------
        df = df.set_index("date").resample("3D").mean().interpolate()
        df = df.reset_index()

        # Convert to API output format
        trends = [
            {"date": row["date"].strftime(
                "%Y-%m-%d"), "value": int(row["value"])}
            for _, row in df.iterrows()
        ]

        anomalies = detect_anomalies(df, "value")

        # ----------------------------------------------
        # CACHE IT
        # ----------------------------------------------
        TRENDS_CACHE[cache_key] = {
            "trends": trends,
            "anomalies": anomalies,
            "fetched_at": datetime.utcnow()
        }

        print(
            f"SerpAPI Google Trends SUCCESS for '{keyword}' — {len(trends)} points")
        return trends, anomalies

    except Exception as e:
        print("Trend fetch failed:", e)

        # fallback to cached if exists
        if cache_key in TRENDS_CACHE:
            return (
                TRENDS_CACHE[cache_key]["trends"],
                TRENDS_CACHE[cache_key]["anomalies"]
            )

        return [], []
