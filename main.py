# main.py
import os
import pyotp
import base64
import qrcode
import asyncio
from io import BytesIO
from fastapi import (
    FastAPI, HTTPException, APIRouter, Depends,
    Response, Cookie, UploadFile, File, Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
from utils.auth_utils import (
    hash_password, verify_password,
    create_access_token, decode_access_token
)
from utils.reddit_api import get_reddit_posts
from utils.category_models import predict_category
from utils.recommendation_engine import router as rec_router
from routes.admin_routes import router as admin_router
from utils.forecasting import multi_model_forecast

load_dotenv()

app = FastAPI()

app.include_router(rec_router)
app.include_router(admin_router)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Static ----------------
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_landing_page():
    path = "static/landing/index.html"
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Landing page not found"}


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
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("SELECT * FROM users WHERE email=%s", (user.email,))
    db_user = cursor.fetchone()
    conn.close()

    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if db_user["role"] == "admin":
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
        secure=os.getenv("ENV") == "production",
        samesite="lax"
    )

    return {"access_token": token}


@app.post("/admin/login")
def admin_login(user: LoginRequest, response: Response):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("SELECT * FROM users WHERE email=%s", (user.email,))
    db_user = cursor.fetchone()
    conn.close()

    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if db_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admins only")

    token = create_access_token({
        "user_id": db_user["id"],
        "email": db_user["email"],
        "role": db_user["role"]
    })

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=os.getenv("ENV") == "production",
        samesite="lax"
    )

    return {"access_token": token}


# ---------------- TRENDS ----------------
class TrendRequest(BaseModel):
    keyword: str
    geo: str = ""


def fetch_and_store_trends(keyword: str, geo: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT date, interest FROM google_trend
        WHERE keyword=%s AND geo=%s ORDER BY date
    """, (keyword, geo))
    rows = cursor.fetchall()

    if rows:
        conn.close()
        return [{"date": r[0].isoformat(), "value": r[1]} for r in rows]

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


@app.get("/forecast-google-trends/{keyword}")
def forecast_trends(keyword: str, region: str | None = None):
    geo = region or ""
    data = fetch_and_store_trends(keyword, geo)

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    result = multi_model_forecast(df)

    return {
        "keyword": keyword,
        "forecast": result["forecast"],
        "chart": result["chart"],
        "insight": result["insight"]
    }


# ---------------- REDDIT ----------------
@app.post("/fetch-reddit/{keyword}")
def fetch_reddit(keyword: str):
    posts = get_reddit_posts(keyword)
    conn = get_connection()
    cursor = conn.cursor()

    for post in posts:
        cursor.execute("""
            INSERT INTO reddit_posts
            (reddit_id, keyword, title, score, url, created_utc, num_comments)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (reddit_id) DO NOTHING
        """, (
            post["reddit_id"],
            keyword,
            post["title"],
            post["score"],
            post["url"],
            post["created_utc"],
            post["num_comments"]
        ))

    conn.commit()
    conn.close()
    return {"status": "ok"}


# ---------------- USER ----------------
def get_current_user(access_token: str | None = Cookie(default=None)):
    if not access_token:
        raise HTTPException(status_code=401)

    payload = decode_access_token(access_token)
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("SELECT * FROM users WHERE id=%s", (payload["user_id"],))
    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=401)

    return user


@app.get("/user/profile")
def user_profile(current_user=Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "name": current_user["name"],
        "email": current_user["email"]
    }



class ChangePwdReq(BaseModel):
    current: str
    newPassword: str


@app.post("/user/change-password")
def change_password(req: ChangePwdReq, current_user=Depends(get_current_user)):
    conn = get_connection()
    if not conn:
        raise HTTPException(
            status_code=500, detail="Database connection failed")
    cursor = conn.cursor(dictionary=True)
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
        cursor = conn.cursor()
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
        cursor = conn.cursor()
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
            cursor = conn.cursor()
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
