import json
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

CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id INTEGER NOT NULL REFERENCES watches(id),
    run_id TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    price REAL NOT NULL,
    currency TEXT NOT NULL,
    confidence TEXT NOT NULL,
    sources TEXT,
    verdict TEXT NOT NULL,
    pct_vs_target REAL NOT NULL,
    trend_pct REAL,
    reasoning TEXT
);

CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
);
"""

# Columns added after the first release: (table, column, DDL type).
_ADDED_COLUMNS = [
    ("watches", "last_alerted_at", "TEXT"),
    ("watches", "last_alerted_price", "REAL"),
    ("price_history", "run_id", "TEXT"),
    ("price_history", "source", "TEXT"),
]


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
        for table, column, ddl_type in _ADDED_COLUMNS:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_watches_identity "
            "ON watches (brand, model, COALESCE(reference_no, ''))"
        )


def add_watch(brand: str, model: str, target_price: float, reference_no: str = "",
              currency: str = "USD", notes: str = "") -> int:
    if target_price <= 0:
        raise ValueError("target price must be positive")
    try:
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO watches (brand, model, reference_no, target_price, currency, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (brand, model, reference_no, target_price, currency, notes),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError as e:
        raise ValueError(f"{brand} {model} ({reference_no or 'no ref'}) is already on the watchlist") from e


def list_watches() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM watches ORDER BY id").fetchall()


def get_watch(watch_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM watches WHERE id = ?", (watch_id,)).fetchone()


def record_price(watch_id: int, price: float, currency: str,
                  source_url: str = "", snippet: str = "",
                  run_id: str = "", source: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO price_history "
            "(watch_id, price, currency, source_url, snippet, fetched_at, run_id, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (watch_id, price, currency, source_url, snippet,
             datetime.now(timezone.utc).isoformat(), run_id, source),
        )
        return cur.lastrowid


def record_check(watch_id: int, run_id: str, price: float, currency: str,
                 confidence: str, sources: str, verdict: str, pct_vs_target: float,
                 trend_pct: float | None, reasoning: str) -> int:
    """Store the outcome of one check: the price used and the verdict computed from it."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO checks (watch_id, run_id, checked_at, price, currency, confidence, "
            "sources, verdict, pct_vs_target, trend_pct, reasoning) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (watch_id, run_id, datetime.now(timezone.utc).isoformat(), price, currency,
             confidence, sources, verdict, pct_vs_target, trend_pct, reasoning),
        )
        return cur.lastrowid


def prior_check_prices(watch_id: int, limit: int = 5) -> list[float]:
    """Prices from earlier checks of this watch, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT price FROM checks WHERE watch_id = ? ORDER BY checked_at DESC, id DESC LIMIT ?",
            (watch_id, limit),
        ).fetchall()
    return [r["price"] for r in rows]


def latest_check(watch_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM checks WHERE watch_id = ? ORDER BY checked_at DESC, id DESC LIMIT 1",
            (watch_id,),
        ).fetchone()


def get_price_history(watch_id: int, limit: int = 20) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM price_history WHERE watch_id = ? ORDER BY fetched_at DESC LIMIT ?",
            (watch_id, limit),
        ).fetchall()


def update_target(watch_id: int, target_price: float) -> bool:
    if target_price <= 0:
        raise ValueError("target price must be positive")
    with get_conn() as conn:
        cur = conn.execute("UPDATE watches SET target_price = ? WHERE id = ?", (target_price, watch_id))
        return cur.rowcount > 0


def remove_watch(watch_id: int) -> bool:
    """Delete a watch and everything recorded about it."""
    with get_conn() as conn:
        conn.execute("DELETE FROM checks WHERE watch_id = ?", (watch_id,))
        conn.execute("DELETE FROM price_history WHERE watch_id = ?", (watch_id,))
        cur = conn.execute("DELETE FROM watches WHERE id = ?", (watch_id,))
        return cur.rowcount > 0


def recent_checks(watch_id: int, limit: int = 10) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM checks WHERE watch_id = ? ORDER BY checked_at DESC, id DESC LIMIT ?",
            (watch_id, limit),
        ).fetchall()


def get_state(key: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def set_state(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO bot_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def create_pending(action: str, payload: dict) -> int:
    """Stage an action that waits for the user's Confirm tap."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO pending_actions (action, payload, created_at) VALUES (?, ?, ?)",
            (action, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def resolve_pending(pending_id: int, status: str) -> tuple[str, dict, str] | None:
    """Move a still-pending action to `status` and return (action, payload, created_at).

    Returns None if it doesn't exist or was already resolved, so a repeated tap
    or a replayed update can never run an action twice.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM pending_actions WHERE id = ? AND status = 'pending'", (pending_id,)
        ).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE pending_actions SET status = ? WHERE id = ?", (status, pending_id))
        return row["action"], json.loads(row["payload"]), row["created_at"]


def mark_alerted(watch_id: int, price: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE watches SET last_alerted_at = ?, last_alerted_price = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), price, watch_id),
        )
