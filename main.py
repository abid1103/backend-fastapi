# main.py
import os
import pyotp
import base64
import qrcode
import asyncio
from io import BytesIO
from fastapi import (
    FastAPI, HTTPException, APIRouter, Depends,
    Response, Cookie, UploadFile, File, Form, Body, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import traceback

from psycopg2.extras import RealDictCursor
from utils.db_connection import get_connection
from utils.google_trends import fetch_google_trends
from utils.sentiment_analysis import analyze_sentiment
from utils.auth_utils import hash_password, verify_password, create_access_token, decode_access_token
from utils.reddit_api import get_reddit_posts, get_comments_for_post
from utils.news_api import fetch_newsapi_posts
from utils.cryptopanic_api import fetch_cryptopanic_posts
from routes.admin_routes import router as admin_router

load_dotenv()

app = FastAPI(root_path="/api")  # All routes prefixed with /api


FRONTEND_URL = os.getenv("FRONTEND_URL", "https://brandinsightai.vercel.app")

# ---------------- Lazy import for deployment blocker ----------------
multi_model_forecast = None
def get_forecast_module():
    global multi_model_forecast
    if multi_model_forecast is None:
        from utils.forecasting import multi_model_forecast
    return multi_model_forecast

# ---------------- Routers ----------------
from utils.recommendation_engine import router as rec_router

app.include_router(rec_router)
app.include_router(admin_router)



# Category → API mapping
CATEGORY_API_MAPPING = {
    "Sports": ["reddit"],
    "Stocks": ["newsapi"],
    "Crypto": ["cryptopanic"],
    "Tech": ["newsapi", "reddit"],
    "Politics": ["newsapi", "reddit"],
    "Entertainment": ["newsapi", "reddit"],
    "other": ["reddit"]
}


# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://brandinsightai.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files ---
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- Landing Page ----------------
@app.get("/")
def serve_landing_page():
    path = "static/landing/index.html"
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Landing page not found"}



@app.get("/dashboard")
@app.get("/dashboard/{path:path}")

def serve_dashboard(path: str = ""):
    return RedirectResponse(f"{FRONTEND_URL}/dashboard/{path}")

# ---------------- Schemas ----------------
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    company: str | None = None

class LoginRequest(BaseModel):
    email: str
    password: str

# ---------------- AUTH ----------------
@app.post("/signup")
def signup(user: SignupRequest):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT id FROM users WHERE email=%s", (user.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    cursor.execute(
        "INSERT INTO users (name, email, company, password) VALUES (%s,%s,%s,%s)",
        (user.name, user.email, user.company, hash_password(user.password))
    )
    conn.commit()
    conn.close()
    return {"message": "Account created successfully"}

@app.post("/login")
def login(user: LoginRequest, response: Response):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM users WHERE email=%s", (user.email,))
    db_user = cursor.fetchone()
    conn.close()

    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if db_user.get("role") == "admin":
        raise HTTPException(status_code=403, detail="Admins cannot login here")

    token = create_access_token({
        "user_id": db_user["id"],
        "email": db_user["email"],
        "role": db_user["role"]
    })
    
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="none"
    )


    return {"access_token": token}

    
@app.post("/admin/login")
def admin_login(user: LoginRequest, response: Response):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM users WHERE email=%s", (user.email,))
    db_user = cursor.fetchone()
    conn.close()

    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if db_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admins only")

    token = create_access_token({
        "user_id": db_user["id"],
        "email": db_user["email"],
        "role": db_user["role"]
    })

    secure_flag = os.getenv("ENV") == "production"
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=secure_flag,
        samesite="lax"
    )

    return {"access_token": token}

    
# --- LOGOUT ---


@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out"}


# ---------------- Google Trends ----------------
class TrendRequest(BaseModel):
    keyword: str
    geo: str = ""

def fetch_and_store_trends(keyword: str, geo: str):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT date, interest FROM google_trend
        WHERE keyword=%s AND geo=%s ORDER BY date
    """, (keyword, geo))
    rows = cursor.fetchall()

    if rows:
        conn.close()
        return [{"date": r["date"].isoformat(), "value": r["interest"]} for r in rows]

    trends, _ = fetch_google_trends(keyword, geo)
    for r in trends:
        cursor.execute("""
            INSERT INTO google_trend (keyword, geo, date, interest)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (keyword, geo, date)
            DO UPDATE SET interest = EXCLUDED.interest
        """, (keyword, geo, r["date"], r["value"]))

    conn.commit()
    conn.close()
    return trends


# -------------------------
# Fetch fresh Google Trends (with DB cache check)
# -------------------------
@app.post("/fetch-trends")
async def fetch_trends_endpoint(body: TrendRequest):
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor) if conn else None

        # Check if data exists and is fresh (<=30 days old)
        fresh_threshold = datetime.utcnow() - timedelta(days=14)
        trends = []
        if cursor:
            cursor.execute("""
                SELECT date, interest FROM google_trend
                WHERE keyword=%s AND geo=%s AND date >= %s
                ORDER BY date
            """, (body.keyword, body.geo, fresh_threshold))
            rows = cursor.fetchall()
            trends = [{"date": row[0].isoformat(), "value": row[1]}
                      for row in rows]

        # If no fresh data, fetch from Google
        if not trends:
            trends, anomalies = fetch_google_trends(body.keyword, body.geo)

            # Save to DB
            if cursor:
                for row in trends:
                    cursor.execute("""
                        INSERT INTO google_trend (keyword, geo, date, interest)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (keyword, geo, date)
                        DO UPDATE SET interest = EXCLUDED.interest
                    """, (body.keyword, body.geo, row["date"], row["value"]))
                conn.commit()

        if conn:
            conn.close()

        return {"status": "ok", "keyword": body.keyword, "trends": trends}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trends")
async def get_trends_endpoint(body: TrendRequest):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            raise HTTPException(
                status_code=500, detail="Database connection failed")

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT date, interest FROM google_trend
            WHERE keyword=%s AND geo=%s
            ORDER BY date
        """, (body.keyword, body.geo))

        rows = cursor.fetchall()

        trends = [{"date": row[0].isoformat(), "value": row[1]}
                  for row in rows]

        anomalies = []  # you already had this

        # ✅ DO NOT CRASH if empty
        return {
            "keyword": body.keyword,
            "trends": trends,
            "anomalies": anomalies
        }

    except HTTPException:
        # ✅ Preserve real HTTP errors (404, 401, etc)
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if conn:
            conn.close()


# --- Reddit ---


@app.post("/fetch-reddit/{keyword}")
def fetch_reddit(keyword: str):
    reddit_data = get_reddit_posts(keyword)
    if not reddit_data:
        return {"error": "No Reddit data found"}
    conn = get_connection()
    if conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        for post in reddit_data:
            ts = post["created_utc"]
            if isinstance(ts, datetime):
                created_dt = ts.astimezone(
                    timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            else:
                created_dt = datetime.fromtimestamp(
                    float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
                INSERT INTO reddit_posts
                (reddit_id, keyword, title, score, url, created_utc, num_comments, sentiment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
                ON CONFLICT (reddit_id) DO NOTHING
            """, (
                post["reddit_id"],  # use post["reddit_id"], not sub.id
                keyword,
                post["title"],
                post["score"],
                post["url"],
                post["created_utc"],
                post["num_comments"]
            ))

        conn.commit()
        conn.close()
        return {"message": f"Reddit posts for '{keyword}' saved successfully!"}
    raise HTTPException(status_code=500, detail="Database connection failed")


@app.post("/analyze-reddit-sentiment")
def analyze_reddit_sentiment():
    conn = get_connection()
    if conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT id, title FROM reddit_posts WHERE sentiment IS NULL"
        )
        posts = cursor.fetchall()

        if not posts:
            conn.close()
            return {"error": "No new Reddit posts to analyze"}

        for post in posts:
            # Analyze sentiment using VADER
            result = analyze_sentiment(post["title"])

            # Take the first sentiment label (positive/negative)
            if result["detailed"]:
                sentiment_label = result["detailed"][0]["sentiment"]
            else:
                sentiment_label = None

            # Update only the 'sentiment' column
            cursor.execute("""
                UPDATE reddit_posts
                SET sentiment = %s
                WHERE id = %s
            """, (sentiment_label, post["id"]))

        conn.commit()
        conn.close()
        return {"message": "Sentiment analysis completed for new Reddit posts"}

    return {"error": "Database connection failed"}


@app.get("/sentiment/{keyword}")
def get_sentiment_results(keyword: str):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Fetch posts with sentiment
    cursor.execute("""
        SELECT title, sentiment
        FROM reddit_posts
        WHERE keyword=%s AND sentiment IS NOT NULL
    """, (keyword,))
    posts = cursor.fetchall()
    conn.close()

    detailed = []
    summary = {"positive": 0, "negative": 0}

    for post in posts:
        sentiment_label = post["sentiment"]
        detailed.append({
            "text": post["title"],
            "sentiment": sentiment_label
        })
        if sentiment_label == "positive":
            summary["positive"] += 1
        elif sentiment_label == "negative":
            summary["negative"] += 1

    return {"summary": summary, "detailedSentiment": detailed}

# ---- forecast --------

# -------------------------
# Helper to fetch & store trends if missing
# -------------------------
def fetch_trends_if_missing(keyword: str, geo: str = "") -> None:
    """
    Checks if the keyword exists in DB. If not, calls the existing fetch_trends_endpoint
    to fetch fresh data and store it in DB.
    """

    # Check DB first
    conn = get_connection()
    if not conn:
        raise HTTPException(
            status_code=500, detail="Database connection failed")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT COUNT(*) FROM google_trend
        WHERE keyword=%s AND geo=%s
    """, (keyword, geo))
    exists = cursor.fetchone()[0]
    conn.close()

    if exists:
        return  # Already in DB

    # Call /fetch-trends endpoint logic
    body = TrendRequest(keyword=keyword, geo=geo)
    # fetch_trends_endpoint is async, so run it
    loop = asyncio.get_event_loop()
    loop.create_task(fetch_trends_endpoint(body))



# -------------------------
# Forecast Google Trends
# -------------------------

@app.get("/forecast-google-trends/{keyword}")
def forecast_trends(keyword: str, region: str | None = None):
    geo = region or ""
    data = fetch_and_store_trends(keyword, geo)

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    forecast_module = get_forecast_module()
    result = forecast_module(df)

    return {
        "keyword": keyword,
        "forecast": result["forecast"],
        "chart": result["chart"],
        "insight": result["insight"]
    }


# ----------------------------
# Business Recommendation Logic
# ----------------------------

@rec_router.get("/{keyword}")
def generate_recommendation(keyword: str):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB Connection failed")

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # 1️⃣ Fetch historical trend
    cursor.execute("""
        SELECT date, interest AS value
        FROM google_trend
        WHERE keyword = %s
        ORDER BY date
    """, (keyword,))
    trend_data = cursor.fetchall()

    if not trend_data:
        raise HTTPException(status_code=404, detail="No trend data found")

    # 2️⃣ Prepare DataFrame for forecasting
    df = pd.DataFrame(trend_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # 3️⃣ Run Forecast
    forecast_result = multi_model_forecast(df)
    forecast = forecast_result.get("best_forecast", [])

    if not forecast:
        raise HTTPException(status_code=500, detail="Forecast failed")

    first = forecast[0]["predicted_value"]
    last = forecast[-1]["predicted_value"]
    growth_percent = ((last - first) / first) * 100 if first != 0 else 0

    # 4️⃣ Reddit Interest
    cursor.execute("""
        SELECT COUNT(*) AS total_posts
        FROM reddit_posts
        WHERE keyword = %s
    """, (keyword,))
    reddit_count = cursor.fetchone().get("total_posts", 0)

    # 5️⃣ Business Category
    from utils.category_models import predict_category
    category = predict_category(keyword)

    conn.close()

    # 6️⃣ Decision Engine
    if growth_percent > 15 and reddit_count > 30:
        recommendation = "✅ Strong Market Entry Opportunity"
        action = "Launch product & scale marketing"
        risk = "Low"
    elif 5 < growth_percent <= 15:
        recommendation = "⚠ Moderate Opportunity"
        action = "Run pilot campaign & validate demand"
        risk = "Medium"
    else:
        recommendation = "❌ Weak Market Signal"
        action = "Avoid investment for now"
        risk = "High"

    return {
        "keyword": keyword,
        "category": category,
        "forecast_growth_percent": round(growth_percent, 2),
        "public_interest_posts": reddit_count,
        "business_recommendation": recommendation,
        "suggested_action": action,
        "investment_risk": risk
    }


# ------------------------------------------
# ---------- AUTH COOKIE UTIL & DEPENDS ----
# ------------------------------------------


def get_current_user(access_token: str | None = Cookie(default=None)):
    """
    Dependency: read cookie 'access_token', decode JWT and fetch user from DB.
    Raises HTTPException(401) if not valid.
    """
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = decode_access_token(access_token)
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=401, detail="Invalid token payload")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    conn = get_connection()
    if not conn:
        raise HTTPException(
            status_code=500, detail="Database connection failed")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        "SELECT id, name, email, avatar_url, two_factor_enabled, two_factor_secret FROM users WHERE id=%s", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ------------------------------------------
# ---------- USER ROUTES (protected) -------
# ------------------------------------------
@app.get("/user/profile")
def user_profile(current_user=Depends(get_current_user)):
    """
    Return minimal profile info.
    """
    return {
        "id": current_user["id"],
        "name": current_user["name"],
        "email": current_user["email"],
        "avatar_url": current_user.get("avatar_url") or None,
        "two_factor_enabled": bool(current_user.get("two_factor_enabled")),
    }


@app.put("/user/profile")
def update_profile(name: str = Form(...), current_user=Depends(get_current_user)):
    """
    Update name (multipart/form recommended to allow future file uploads in same form).
    """
    conn = get_connection()
    if not conn:
        raise HTTPException(
            status_code=500, detail="Database connection failed")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("UPDATE users SET name=%s WHERE id=%s",
                   (name, current_user["id"]))
    conn.commit()
    conn.close()
    return {"status": "ok", "name": name}


class ChangePwdReq(BaseModel):
    current: str
    newPassword: str


@app.post("/user/change-password")
def change_password(req: ChangePwdReq, current_user=Depends(get_current_user)):
    conn = get_connection()
    if not conn:
        raise HTTPException(
            status_code=500, detail="Database connection failed")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT password FROM users WHERE id=%s",
                   (current_user["id"],))
    row = cursor.fetchone()
    if not row or not verify_password(req.current, row["password"]):
        raise HTTPException(
            status_code=401, detail="Incorrect current password")
    new_hash = hash_password(req.newPassword)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password=%s WHERE id=%s",
                   (new_hash, current_user["id"]))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# Avatar upload
@app.post("/user/avatar")
def upload_avatar(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    """
    Save uploaded avatar to static/uploads/{user_id}_{filename}
    and update users.avatar_url column.
    """
    try:
        contents = file.file.read()
        ext = os.path.splitext(file.filename)[1] or ".png"
        safe_name = f"user_{current_user['id']}{ext}"
        save_path = os.path.join("static", "uploads", safe_name)
        with open(save_path, "wb") as f:
            f.write(contents)
        # store relative URL
        avatar_url = f"/static/uploads/{safe_name}"

        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("UPDATE users SET avatar_url=%s WHERE id=%s",
                       (avatar_url, current_user["id"]))
        conn.commit()
        conn.close()
        return {"avatar_url": avatar_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# 2FA (TOTP) - basic flow
# ----------------------------
@app.post("/user/2fa/setup")
def twofa_setup(current_user=Depends(get_current_user)):
    """
    Generate TOTP secret and QR image (base64). Save secret to DB as temporary secret (not enabling until verify).
    """
    try:
        secret = pyotp.random_base32()
        # provisioning uri for authenticator apps
        provisioning_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=current_user["email"], issuer_name="BrandInsight")
        # create qr image
        qr = qrcode.make(provisioning_uri)
        buffered = BytesIO()
        qr.save(buffered, format="PNG")
        qr_b64 = base64.b64encode(buffered.getvalue()).decode()
        # Save temp secret to DB column 'two_factor_secret' (you may choose a separate temp column)
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "UPDATE users SET two_factor_secret=%s WHERE id=%s", (secret, current_user["id"]))
        conn.commit()
        conn.close()

        return {"qr_base64": qr_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TwoFAVerify(BaseModel):
    code: str


@app.post("/user/2fa/verify")
def twofa_verify(req: TwoFAVerify, current_user=Depends(get_current_user)):
    """
    Verify the TOTP code; if valid, set two_factor_enabled = True.
    """
    try:
        secret = current_user.get("two_factor_secret")
        if not secret:
            raise HTTPException(
                status_code=400, detail="2FA not initialized for this user")
        totp = pyotp.TOTP(secret)
        if totp.verify(req.code):
            conn = get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "UPDATE users SET two_factor_enabled=1 WHERE id=%s", (current_user["id"],))
            conn.commit()
            conn.close()
            return {"status": "ok"}
        else:
            raise HTTPException(status_code=400, detail="Invalid 2FA code")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
