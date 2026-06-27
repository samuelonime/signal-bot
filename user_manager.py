"""
User Manager — Stores subscriber Deriv tokens and trade settings in PostgreSQL.
"""

import os
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_engine():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/signal_bot")
    return create_engine(db_url, pool_pre_ping=True)


def init_user_tables():
    """Create subscriber tables if not exist."""
    engine = get_engine()
    ddl = """
    CREATE TABLE IF NOT EXISTS subscribers (
        id              SERIAL PRIMARY KEY,
        chat_id         VARCHAR(50) UNIQUE NOT NULL,
        username        VARCHAR(100),
        deriv_token     TEXT,
        trade_amount    NUMERIC(10,2) DEFAULT 10.00,
        auto_trade      BOOLEAN DEFAULT FALSE,
        connected_at    TIMESTAMPTZ,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        updated_at      TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    logger.info("Subscriber tables initialised.")


def save_token(chat_id: str, token: str, username: str = "") -> bool:
    """Save or update a subscriber's Deriv token."""
    engine = get_engine()
    sql = text("""
        INSERT INTO subscribers (chat_id, username, deriv_token, connected_at, updated_at)
        VALUES (:chat_id, :username, :token, NOW(), NOW())
        ON CONFLICT (chat_id) DO UPDATE
            SET deriv_token=EXCLUDED.deriv_token,
                username=EXCLUDED.username,
                connected_at=NOW(),
                updated_at=NOW()
    """)
    try:
        with engine.connect() as conn:
            conn.execute(sql, {"chat_id": chat_id, "username": username, "token": token})
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"save_token error: {exc}")
        return False


def remove_token(chat_id: str) -> bool:
    """Remove a subscriber's Deriv token."""
    engine = get_engine()
    sql = text("""
        UPDATE subscribers
        SET deriv_token=NULL, connected_at=NULL, updated_at=NOW()
        WHERE chat_id=:chat_id
    """)
    try:
        with engine.connect() as conn:
            conn.execute(sql, {"chat_id": chat_id})
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"remove_token error: {exc}")
        return False


def set_amount(chat_id: str, amount: float) -> bool:
    """Set a subscriber's default trade amount."""
    engine = get_engine()
    sql = text("""
        INSERT INTO subscribers (chat_id, trade_amount, updated_at)
        VALUES (:chat_id, :amount, NOW())
        ON CONFLICT (chat_id) DO UPDATE
            SET trade_amount=EXCLUDED.trade_amount, updated_at=NOW()
    """)
    try:
        with engine.connect() as conn:
            conn.execute(sql, {"chat_id": chat_id, "amount": amount})
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"set_amount error: {exc}")
        return False


def get_subscriber(chat_id: str) -> Optional[dict]:
    """Get subscriber info."""
    engine = get_engine()
    sql = text("""
        SELECT chat_id, username, deriv_token, trade_amount, connected_at
        FROM subscribers WHERE chat_id=:chat_id
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"chat_id": chat_id}).fetchone()
        if result:
            return {
                "chat_id":      result[0],
                "username":     result[1],
                "deriv_token":  result[2],
                "trade_amount": float(result[3]) if result[3] else 10.0,
                "connected_at": result[4],
            }
        return None
    except Exception as exc:
        logger.error(f"get_subscriber error: {exc}")
        return None


def get_all_connected() -> list:
    """Get all subscribers with connected Deriv accounts."""
    engine = get_engine()
    sql = text("""
        SELECT chat_id, deriv_token, trade_amount
        FROM subscribers
        WHERE deriv_token IS NOT NULL AND deriv_token != ''
    """)
    try:
        with engine.connect() as conn:
            results = conn.execute(sql).fetchall()
        return [
            {"chat_id": r[0], "deriv_token": r[1], "trade_amount": float(r[2])}
            for r in results
        ]
    except Exception as exc:
        logger.error(f"get_all_connected error: {exc}")
        return []
