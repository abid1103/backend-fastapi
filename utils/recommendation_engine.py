# recommendation_engine.py
import os
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import mysql.connector
from mysql.connector import Error
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from utils.forecasting import multi_model_forecast
from utils.db_connection import get_connection

# load .env
load_dotenv()


router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])

# ---------- Pydantic models ----------


class BatchRequest(BaseModel):
    keywords: List[str]
    geo: Optional[str] = ""
    forecast_days: Optional[int] = 30


class RecommendationOut(BaseModel):
    keyword: str
    score: float
    category: str
    action: str
    reasons: List[str]
    details: Dict[str, Any]
    generated_at: datetime


# ---------- Weights & thresholds ----------
WEIGHTS = {"trend_component": 0.55, "forecast_component": 0.45}
TREND_WEIGHTS = {"trend_avg": 0.5, "direction": 0.3, "spike_intensity": 0.2}
FORECAST_WEIGHTS = {"growth_pct": 0.7, "confidence": 0.3}
THRESHOLDS = {"strong_opportunity": 0.80,
              "opportunity": 0.60, "monitor": 0.40, "reduce": 0.20}

# ---------- Helpers ----------


def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return 0.0


def normalize(value, vmin, vmax):
    if value is None:
        return 0.0
    if vmax == vmin:
        return 0.0
    v = (value - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, v))


def normalize_percent_to_unit(pct, min_pct=-100.0, max_pct=200.0):
    return normalize(pct, min_pct, max_pct)


def direction_score_from_label(label: str) -> float:
    mapping = {"upward": 1.0, "stable": 0.5, "downward": 0.0}
    return mapping.get(label.lower(), 0.5)


def confidence_score_from_label(conf: str) -> float:
    if not conf:
        return 0.5
    conf = conf.lower()
    if conf == "high":
        return 1.0
    if conf == "medium":
        return 0.6
    if conf == "low":
        return 0.3
    return 0.5

# ---------- Fetchers (MySQL) ----------


def fetch_trends_from_db(keyword: str, geo: str = "", days: int = 365) -> List[Dict[str, Any]]:
    """
    Returns list of {date: 'YYYY-MM-DD', value: int}
    Table: google_trend (id, keyword, geo, date (DATE), interest)
    """
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        q = "SELECT date, interest FROM google_trend WHERE keyword = %s"
        params = [keyword]
        if geo:
            q += " AND geo = %s"
            params.append(geo)
        cutoff_date = (datetime.utcnow().date() -
                       timedelta(days=days)).strftime("%Y-%m-%d")
        q += " AND date >= %s ORDER BY date ASC"
        params.append(cutoff_date)
        cur.execute(q, tuple(params))
        rows = cur.fetchall()
        return [{"date": r["date"].strftime("%Y-%m-%d"), "value": int(r["interest"])} for r in rows]
    finally:
        cur.close()
        conn.close()


def fetch_reddit_sentiment_summary(keyword: str, days: int = 90) -> Dict[str, Any]:
    """
    Aggregates counts and simple stats from reddit_posts table:
    reddit_posts: id, keyword, title, score, url, created_utc, num_comments, sentiment
    sentiment expected values: 'positive','negative','neutral'
    """
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)
                  ).strftime("%Y-%m-%d %H:%M:%S")
        q = """
            SELECT sentiment, COUNT(*) AS cnt, AVG(score) AS avg_score, AVG(num_comments) AS avg_comments
            FROM reddit_posts
            WHERE keyword = %s AND created_utc >= %s
            GROUP BY sentiment
        """
        cur.execute(q, (keyword, cutoff))
        rows = cur.fetchall()
        summary = {"total": 0, "positive": 0, "negative": 0,
                   "neutral": 0, "avg_score": 0.0, "avg_comments": 0.0}
        total_score_acc = 0.0
        total_comments_acc = 0.0
        total_posts = 0
        for r in rows:
            s = (r["sentiment"] or "neutral").lower()
            cnt = int(r["cnt"])
            summary[s] = cnt
            total_posts += cnt
            total_score_acc += (r["avg_score"] or 0.0) * cnt
            total_comments_acc += (r["avg_comments"] or 0.0) * cnt
        summary["total"] = total_posts
        if total_posts > 0:
            summary["avg_score"] = total_score_acc / total_posts
            summary["avg_comments"] = total_comments_acc / total_posts
        return summary
    finally:
        cur.close()
        conn.close()

# ---------- Forecast caller ----------


def call_forecast_module(historical_trends: List[Dict[str, Any]], forecast_days: int = 30):
    """
    Calls forecasting.forecast_google_trends(data, look_back, forecast_days)
    Returns {"forecast": [...], "chart": base64}
    """
    try:
        forecast_result, chart_base64 = forecast_google_trends(
            historical_trends, look_back=30, forecast_days=forecast_days)
        return {"forecast": forecast_result, "chart": chart_base64}
    except Exception:
        return {"forecast": [], "chart": ""}

# ---------- Anomaly detection ----------


def detect_anomalies_from_values(trends_sorted: List[Dict[str, Any]]):
    import pandas as pd
    if not trends_sorted:
        return []
    df = pd.DataFrame(trends_sorted).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    if df.empty:
        return []
    values = df["value"]
    mean = values.mean()
    std = values.std() if values.std() != 0 else 1.0
    z = (values - mean) / std
    rolling = values.rolling(window=7, min_periods=3).mean()
    anomalies = []
    for idx, row in df.iterrows():
        spike = False
        if abs(z.iloc[idx]) > 2.5:
            spike = True
        if not spike and idx > 0 and not pd.isna(rolling.iloc[idx]):
            if row["value"] > rolling.iloc[idx] * 1.4:
                spike = True
        if spike:
            anomalies.append({"date": row["date"].strftime("%Y-%m-%d")})
    return anomalies

# ---------- Metric computations ----------


def compute_trend_metrics(trends_list: List[Dict[str, Any]], anomalies: List[Dict[str, Any]]):
    if not trends_list:
        return {
            "trend_avg": 0.0,
            "direction_label": "stable",
            "direction_score": 0.5,
            "spike_count_30d": 0,
            "spike_intensity": 0.0,
            "recency_days": None,
            "volume_avg": 0.0
        }
    trends_sorted = sorted(trends_list, key=lambda x: x["date"])
    values = [float(x["value"]) for x in trends_sorted]
    trend_avg = sum(values) / len(values)
    n = len(values)
    window = max(1, math.floor(n * 0.3))
    recent_avg = sum(values[-window:]) / window
    initial_avg = sum(values[:window]) / window
    pct_change = safe_div((recent_avg - initial_avg),
                          max(1.0, initial_avg)) * 100.0
    if pct_change > 10:
        direction_label = "upward"
    elif pct_change < -10:
        direction_label = "downward"
    else:
        direction_label = "stable"
    direction_score = direction_score_from_label(direction_label)
    now = datetime.utcnow().date()
    spike_dates = []
    for a in anomalies:
        try:
            d = datetime.strptime(a.get("date"), "%Y-%m-%d").date()
            spike_dates.append(d)
        except Exception:
            pass
    spike_dates = sorted(spike_dates)
    cutoff = now - timedelta(days=30)
    spikes_30d = sum(1 for d in spike_dates if d >= cutoff)
    recency_days = (now - spike_dates[-1]).days if spike_dates else None
    spike_intensity = min(1.0, spikes_30d / 5.0)
    volume_avg = trend_avg
    return {
        "trend_avg": trend_avg,
        "direction_label": direction_label,
        "direction_score": direction_score,
        "spike_count_30d": spikes_30d,
        "spike_intensity": spike_intensity,
        "recency_days": recency_days,
        "volume_avg": volume_avg
    }


def compute_forecast_metrics(forecast_list: List[Dict[str, Any]]):
    if not forecast_list or len(forecast_list) < 2:
        return {"growth_pct": 0.0, "confidence_label": "low", "predicted_avg": 0.0, "first_pred": 0.0, "last_pred": 0.0}
    first = float(forecast_list[0]["predicted_value"])
    last = float(forecast_list[-1]["predicted_value"])
    avg_pred = sum([float(x["predicted_value"])
                   for x in forecast_list]) / len(forecast_list)
    denom = first if first != 0 else 1.0
    pct = (last - first) / denom * 100.0
    conf = "medium"
    if len(forecast_list) >= 30:
        conf = "high"
    elif len(forecast_list) < 10:
        conf = "low"
    return {"growth_pct": pct, "confidence_label": conf, "predicted_avg": avg_pred, "first_pred": first, "last_pred": last}


def compute_scores_from_metrics(trend_metrics: dict, forecast_metrics: dict, sentiment_metrics: dict = None):
    t_avg_norm = normalize(trend_metrics.get("trend_avg", 0.0), 0.0, 100.0)
    t_dir = trend_metrics.get("direction_score", 0.5)
    t_spike = trend_metrics.get("spike_intensity", 0.0)
    trend_component = (TREND_WEIGHTS["trend_avg"] * t_avg_norm +
                       TREND_WEIGHTS["direction"] * t_dir +
                       TREND_WEIGHTS["spike_intensity"] * t_spike)
    f_growth_norm = normalize_percent_to_unit(
        forecast_metrics.get("growth_pct", 0.0), -100.0, 200.0)
    f_conf = confidence_score_from_label(
        forecast_metrics.get("confidence_label", "medium"))
    forecast_component = (FORECAST_WEIGHTS["growth_pct"] * f_growth_norm +
                          FORECAST_WEIGHTS["confidence"] * f_conf)
    sentiment_component = 0.5
    sentiment_impact = 0.0
    if sentiment_metrics:
        total = sentiment_metrics.get("total", 0)
        pos = sentiment_metrics.get("positive", 0)
        neg = sentiment_metrics.get("negative", 0)
        if total > 0:
            net = safe_div((pos - neg), total)  # -1..1
            sentiment_component = (net + 1.0) / 2.0
            sentiment_impact = sentiment_component - 0.5
    final_score = WEIGHTS["trend_component"] * trend_component + \
        WEIGHTS["forecast_component"] * forecast_component
    final_score = final_score + (0.15 * sentiment_impact)
    final_score = max(0.0, min(1.0, round(final_score, 4)))
    return {
        "trend_component": round(trend_component, 4),
        "forecast_component": round(forecast_component, 4),
        "sentiment_component": round(sentiment_component, 4),
        "final_score": final_score,
        "t_avg_norm": round(t_avg_norm, 4),
        "f_growth_norm": round(f_growth_norm, 4)
    }

# ---------- Business mapping ----------


def map_score_to_action(score: float, trend_metrics: dict, forecast_metrics: dict, sentiment_metrics: dict = None):
    reasons = []
    if forecast_metrics.get("growth_pct", 0.0) <= -15.0:
        reasons.append("Forecast predicts significant decline")
        return "Reduce / De-prioritize", "High Risk", reasons
    if trend_metrics.get("spike_intensity", 0.0) >= 0.8 and (trend_metrics.get("recency_days") is not None and trend_metrics.get("recency_days") <= 7):
        reasons.append(
            "Recent high-intensity spikes detected — investigate sentiment or news")
        return "Investigate / Monitor closely", "Monitor", reasons
    if sentiment_metrics and sentiment_metrics.get("total", 0) > 0:
        neg = sentiment_metrics.get("negative", 0)
        total = sentiment_metrics.get("total", 1)
        if safe_div(neg, total) >= 0.4:
            reasons.append("High negative sentiment in discussions")
            return "Investigate / Pause promotion", "Monitor", reasons
    if score >= THRESHOLDS["strong_opportunity"]:
        reasons.append(
            "High composite score from trend, forecast and sentiment")
        return "Stock up & Promote", "Strong Opportunity", reasons
    if score >= THRESHOLDS["opportunity"]:
        reasons.append(
            "Good composite score; consider marketing & inventory prep")
        return "Increase marketing & Prepare inventory", "Opportunity", reasons
    if score >= THRESHOLDS["monitor"]:
        reasons.append("Moderate score; monitor trend and forecast")
        return "Monitor / Consider small campaign", "Monitor", reasons
    if score >= THRESHOLDS["reduce"]:
        reasons.append("Low score; low priority")
        return "Reduce budget / Hold", "Reduce", reasons
    reasons.append("Very low score; avoid or divest")
    return "Avoid / De-prioritize", "High Risk", reasons

# ---------- Persist to DB ----------


def save_recommendation_to_db(rec: RecommendationOut):
    conn = get_connection()
    cur = conn.cursor()
    try:
        q = """INSERT INTO recommendations (keyword, score, category, action, reasons, details, generated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        cur.execute(q, (
            rec.keyword,
            float(rec.score),
            rec.category,
            rec.action,
            json.dumps(rec.reasons),
            json.dumps(rec.details),
            rec.generated_at.strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
    finally:
        cur.close()
        conn.close()

# ---------- Endpoints ----------


@router.post("/batch", response_model=List[RecommendationOut])
async def generate_batch(req: BatchRequest):
    if not req.keywords:
        raise HTTPException(status_code=400, detail="keywords required")
    results: List[RecommendationOut] = []
    for kw in req.keywords:
        try:
            trends = fetch_trends_from_db(kw, req.geo, days=365)
            anomalies = detect_anomalies_from_values(trends)
            trend_metrics = compute_trend_metrics(trends, anomalies)
            forecast_resp = call_forecast_module(
                trends, forecast_days=req.forecast_days or 30)
            forecast_list = forecast_resp.get("forecast", [])
            forecast_metrics = compute_forecast_metrics(forecast_list)
            sentiment_summary = fetch_reddit_sentiment_summary(kw, days=90)
            score_breakdown = compute_scores_from_metrics(
                trend_metrics, forecast_metrics, sentiment_summary)
            action, category, reasons = map_score_to_action(
                score_breakdown["final_score"], trend_metrics, forecast_metrics, sentiment_summary)
            out = RecommendationOut(
                keyword=kw,
                score=score_breakdown["final_score"],
                category=category,
                action=action,
                reasons=reasons,
                details={
                    "trend": trend_metrics,
                    "forecast": forecast_metrics,
                    "sentiment": sentiment_summary,
                    "score_breakdown": score_breakdown,
                    "forecast_chart": forecast_resp.get("chart", "")
                },
                generated_at=datetime.utcnow()
            )
            # persist
            try:
                save_recommendation_to_db(out)
            except Exception:
                # do not fail entire batch if save fails
                pass
            results.append(out)
        except Exception as e:
            results.append(RecommendationOut(
                keyword=kw,
                score=0.0,
                category="Error",
                action="--",
                reasons=[f"Processing failed: {str(e)}"],
                details={},
                generated_at=datetime.utcnow()
            ))
    results_sorted = sorted(results, key=lambda x: x.score if hasattr(
        x, "score") else 0.0, reverse=True)
    return results_sorted


@router.get("/keyword/{keyword}", response_model=RecommendationOut)
async def recommend_for_keyword(keyword: str, geo: Optional[str] = "", forecast_days: int = 30):
    batch = BatchRequest(keywords=[keyword],
                         geo=geo, forecast_days=forecast_days)
    res = await generate_batch(batch)
    return res[0]


@router.get("/list")
async def list_recommendations(limit: int = 50):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        q = "SELECT id, keyword, score, category, action, reasons, details, generated_at FROM recommendations ORDER BY generated_at DESC LIMIT %s"
        cur.execute(q, (limit,))
        rows = cur.fetchall()
        for r in rows:
            # convert JSON fields if stored as text
            try:
                r["reasons"] = json.loads(r["reasons"]) if r["reasons"] else []
            except Exception:
                r["reasons"] = r["reasons"]
            try:
                r["details"] = json.loads(r["details"]) if r["details"] else {}
            except Exception:
                r["details"] = r["details"]
        return rows
    finally:
        cur.close()
        conn.close()


@router.get("/download")
async def download_csv(keyword: Optional[str] = None):
    """
    Returns CSV of recommendations filtered by keyword (if provided), else all.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        if keyword:
            q = "SELECT keyword, score, category, action, reasons, details, generated_at FROM recommendations WHERE keyword = %s ORDER BY generated_at DESC"
            cur.execute(q, (keyword,))
        else:
            q = "SELECT keyword, score, category, action, reasons, details, generated_at FROM recommendations ORDER BY generated_at DESC"
            cur.execute(q)
        rows = cur.fetchall()
        # build CSV
        import io
        import csv
        sio = io.StringIO()
        writer = csv.writer(sio)
        writer.writerow(["keyword", "score", "category",
                        "action", "reasons", "details", "generated_at"])
        for r in rows:
            kw, score, cat, action, reasons, details, generated_at = r
            writer.writerow([kw, float(score), cat, action, reasons, details, generated_at.strftime(
                "%Y-%m-%d %H:%M:%S") if isinstance(generated_at, datetime) else generated_at])
        csv_bytes = sio.getvalue().encode("utf-8")
        return Response(content=csv_bytes, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=recommendations_{datetime.utcnow().strftime('%Y%m%d%H%M')}.csv"})
    finally:
        cur.close()
        conn.close()
