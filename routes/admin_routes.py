from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import secrets

from utils.db_connection import get_connection
from utils.auth_utils import decode_access_token, hash_password

router = APIRouter(prefix="/admin", tags=["admin"])


def fetch_user_by_id(cursor, user_id: int):
    cursor.execute(
        "SELECT id, name, email, company, created_at, avatar_url, two_factor_enabled, role FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone()


def is_user_admin(db_user: Dict) -> bool:
    if not db_user:
        return False
    role = db_user.get("role")
    if role:
        return role == "admin"
    admin_emails = os.getenv("ADMIN_EMAILS", "")
    if admin_emails:
        admin_list = [e.strip().lower()
                      for e in admin_emails.split(",") if e.strip()]
        if db_user.get("email", "").lower() in admin_list:
            return True
    return db_user.get("id") == 1


async def get_current_admin(request: Request):
    # 1) Try cookie
    token = request.cookies.get("access_token")

    # 2) If no cookie, try Authorization header
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_access_token(token)
    if not payload or "user_id" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user_id = payload["user_id"]

    conn = get_connection()
    if not conn:
        raise HTTPException(
            status_code=500, detail="Database connection failed")
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    current_user = cursor.fetchone()
    conn.close()

    if not current_user:
        raise HTTPException(status_code=401, detail="User not found")
    if not is_user_admin(current_user):
        raise HTTPException(
            status_code=403, detail="Admin privileges required")
    return current_user


class UpdateUserRequest(BaseModel):
    name: Optional[str]
    email: Optional[str]
    company: Optional[str]
    role: Optional[str]  # 'user' or 'admin'


class ResetPasswordRequest(BaseModel):
    new_password: Optional[str] = None


@router.get("/users", response_model=List[dict])
def list_users(limit: int = 50, offset: int = 0, admin=Depends(get_current_admin)):
    conn = get_connection()
    if not conn:
        raise HTTPException(
            status_code=500, detail="Database connection failed")
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, name, email, company, created_at, avatar_url, two_factor_enabled, role FROM users ORDER BY id DESC LIMIT %s OFFSET %s",
        (limit, offset)
    )
    users = cursor.fetchall()
    conn.close()
    return users


@router.get("/users/{user_id}")
def get_user(user_id: int, admin=Depends(get_current_admin)):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    user = fetch_user_by_id(cursor, user_id)
    conn.close()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/users/{user_id}")
def update_user(user_id: int, payload: UpdateUserRequest, admin=Depends(get_current_admin)):
    conn = get_connection()
    cursor = conn.cursor()
    fields = []
    values = []
    if payload.name is not None:
        fields.append("name = %s")
        values.append(payload.name)
    if payload.email is not None:
        fields.append("email = %s")
        values.append(payload.email)
    if payload.company is not None:
        fields.append("company = %s")
        values.append(payload.company)
    if payload.role is not None:
        if payload.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="Invalid role")
        fields.append("role = %s")
        values.append(payload.role)
    if not fields:
        conn.close()
        raise HTTPException(status_code=400, detail="No fields to update")
    values.append(user_id)
    sql = f"UPDATE users SET {', '.join(fields)} WHERE id = %s"
    cursor.execute(sql, tuple(values))
    conn.commit()
    conn.close()
    return {"message": "User updated"}


@router.delete("/users/{user_id}")
def delete_user(user_id: int, admin=Depends(get_current_admin)):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted"}


@router.post("/users/{user_id}/reset-password")
def reset_password(user_id: int, body: ResetPasswordRequest, admin=Depends(get_current_admin)):
    new_pw = body.new_password if body.new_password else secrets.token_urlsafe(
        8)
    hashed = hash_password(new_pw)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET password = %s WHERE id = %s", (hashed, user_id))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Password reset", "temporary_password": new_pw}


@router.post("/users/{user_id}/toggle-2fa")
def toggle_2fa(user_id: int, enable: bool, admin=Depends(get_current_admin)):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET two_factor_enabled = %s WHERE id = %s",
                   (1 if enable else 0, user_id))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"2FA {'enabled' if enable else 'disabled'} for user {user_id}"}
