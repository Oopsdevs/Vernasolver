"""Lightweight user accounts: SQLite + PBKDF2 + session tokens."""
import hashlib
import re
import secrets
import sqlite3
import time
from pathlib import Path

from config import DB_DIR

USERS_DB = DB_DIR / "users.db"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PBKDF2_ITERS = 120_000


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(USERS_DB)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                name          TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    INTEGER NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)


def _hash_password(password: str, salt: bytes | None = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return f"{salt.hex()}:{h.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        new_h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
        return secrets.compare_digest(new_h.hex(), hash_hex)
    except Exception:
        return False


def _validate(email: str, name: str, password: str) -> None:
    email = email.strip().lower()
    name = name.strip()
    if not EMAIL_RE.match(email):
        raise ValueError("Please enter a valid email address")
    if len(name) < 2:
        raise ValueError("Name must be at least 2 characters")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")


def create_user(email: str, name: str, password: str) -> dict:
    email = email.strip().lower()
    name = name.strip()
    _validate(email, name, password)
    stored = _hash_password(password)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (email, name, stored, int(time.time())),
        )
        return {"id": cur.lastrowid, "email": email, "name": name}


def authenticate(email: str, password: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT id, email, name, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    if not row or not _verify_password(password, row[3]):
        return None
    return {"id": row[0], "email": row[1], "name": row[2]}


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, int(time.time())),
        )
    return token


def get_user_by_token(token: str) -> dict | None:
    if not token:
        return None
    with _conn() as c:
        row = c.execute(
            """
            SELECT u.id, u.email, u.name
            FROM sessions s JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
    if not row:
        return None
    return {"id": row[0], "email": row[1], "name": row[2]}


def delete_session(token: str) -> None:
    if not token:
        return
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


init_db()
