import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from config import DB_PATH


def init_db():
    """Buat tabel kalau belum ada. Dipanggil sekali saat app start."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS zones (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                points    TEXT NOT NULL,   -- JSON: [[x,y], ...]
                direction TEXT NOT NULL DEFAULT 'both',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS crossings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                zone_id    INTEGER NOT NULL,
                person_id  INTEGER NOT NULL,
                event      TEXT NOT NULL,  -- 'enter' atau 'exit'
                timestamp  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (zone_id)    REFERENCES zones(id)
            );
        """)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Sessions ──────────────────────────────────────────────────────────────────

def start_session() -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)",
            (datetime.now().isoformat(),)
        )
        return cur.lastrowid


def end_session(session_id: int):
    with _conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (datetime.now().isoformat(), session_id)
        )


# ── Zones ─────────────────────────────────────────────────────────────────────

def save_zone(name: str, points: list, direction: str = "both") -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO zones (name, points, direction, created_at) VALUES (?, ?, ?, ?)",
            (name, json.dumps(points), direction, datetime.now().isoformat())
        )
        return cur.lastrowid


def update_zone(zone_id: int, name: str, points: list, direction: str):
    with _conn() as conn:
        conn.execute(
            "UPDATE zones SET name=?, points=?, direction=? WHERE id=?",
            (name, json.dumps(points), direction, zone_id)
        )


def delete_zone(zone_id: int):
    with _conn() as conn:
        conn.execute("DELETE FROM zones WHERE id=?", (zone_id,))
        conn.execute("DELETE FROM crossings WHERE zone_id=?", (zone_id,))


def load_all_zones() -> list[dict]:
    """Load semua zone dari DB — dipanggil saat app start untuk restore state."""
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM zones").fetchall()
        return [
            {
                "zone_id":   row["id"],
                "name":      row["name"],
                "points":    json.loads(row["points"]),
                "direction": row["direction"],
            }
            for row in rows
        ]


# ── Crossings ─────────────────────────────────────────────────────────────────

def log_crossing(session_id: int, zone_id: int,
                 person_id: int, event: str):
    with _conn() as conn:
        conn.execute(
            """INSERT INTO crossings
               (session_id, zone_id, person_id, event, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, zone_id, person_id, event,
             datetime.now().isoformat())
        )


def get_crossings(zone_id: int | None = None,
                  session_id: int | None = None,
                  event: str | None = None) -> list[dict]:
    query  = "SELECT * FROM crossings WHERE 1=1"
    params = []
    if zone_id:
        query += " AND zone_id=?";    params.append(zone_id)
    if session_id:
        query += " AND session_id=?"; params.append(session_id)
    if event:
        query += " AND event=?";      params.append(event)
    query += " ORDER BY timestamp DESC"

    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_zone_stats(session_id: int | None = None) -> list[dict]:
    """Aggregasi count_in dan count_out per zone."""
    session_filter = "AND session_id=?" if session_id else ""
    params = [session_id] if session_id else []

    query = f"""
        SELECT
            z.id   AS zone_id,
            z.name AS name,
            SUM(CASE WHEN c.event='enter' THEN 1 ELSE 0 END) AS count_in,
            SUM(CASE WHEN c.event='exit'  THEN 1 ELSE 0 END) AS count_out
        FROM zones z
        LEFT JOIN crossings c ON z.id = c.zone_id {session_filter}
        GROUP BY z.id
    """
    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    

def get_heatmap_stats() -> list[dict]:
    """Calculate visit frequency and average dwell time per zone."""
    query = """
        SELECT 
            z.id AS zone_id,
            z.name,
            COUNT(CASE WHEN c1.event = 'enter' THEN 1 END) AS total_visits,
            AVG(strftime('%s', c2.timestamp) - strftime('%s', c1.timestamp)) AS avg_dwell_seconds
        FROM zones z
        LEFT JOIN crossings c1 ON z.id = c1.zone_id AND c1.event = 'enter'
        LEFT JOIN crossings c2 ON z.id = c2.zone_id AND c2.person_id = c1.person_id 
            AND c2.event = 'exit' AND c2.timestamp > c1.timestamp
        GROUP BY z.id
    """
    with _conn() as conn:
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
