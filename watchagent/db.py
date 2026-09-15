import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from watchagent.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS watches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    reference_no TEXT,
    target_price REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    notes TEXT,
    last_alerted_at TEXT
);

CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id INTEGER NOT NULL REFERENCES watches(id),
    price REAL NOT NULL,
    currency TEXT NOT NULL,
    source_url TEXT,
    snippet TEXT,
    fetched_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(watches)")}
        if "last_alerted_at" not in columns:
            conn.execute("ALTER TABLE watches ADD COLUMN last_alerted_at TEXT")


def add_watch(brand: str, model: str, target_price: float, reference_no: str = "",
              currency: str = "USD", notes: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO watches (brand, model, reference_no, target_price, currency, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (brand, model, reference_no, target_price, currency, notes),
        )
        return cur.lastrowid


def list_watches() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM watches ORDER BY id").fetchall()


def get_watch(watch_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM watches WHERE id = ?", (watch_id,)).fetchone()


def record_price(watch_id: int, price: float, currency: str,
                  source_url: str = "", snippet: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO price_history (watch_id, price, currency, source_url, snippet, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (watch_id, price, currency, source_url, snippet,
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_price_history(watch_id: int, limit: int = 20) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM price_history WHERE watch_id = ? ORDER BY fetched_at DESC LIMIT ?",
            (watch_id, limit),
        ).fetchall()


def mark_alerted(watch_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE watches SET last_alerted_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), watch_id),
        )
