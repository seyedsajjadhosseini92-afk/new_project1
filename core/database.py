"""
Thread-safe SQLite database layer.

Tables
------
people  — registered person registry (id, name, added timestamp)
visits  — entry / exit events per person
seen    — periodic "still visible" log per person
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional

from core.config import CONFIG


class Database:
    """Wraps SQLite with a per-thread connection and a write lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_schema()
        self._migrate_v2()
        self._migrate_v1()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(CONFIG.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _now(self) -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _init_schema(self):
        conn = sqlite3.connect(CONFIG.db_path)
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS people (
                id          TEXT PRIMARY KEY,
                name        TEXT,
                department  TEXT,
                employee_id TEXT,
                role        TEXT DEFAULT 'employee',
                added       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS visits (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                action    TEXT NOT NULL,
                time      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS seen (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                time      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_visits_pid ON visits(person_id);
            CREATE INDEX IF NOT EXISTS idx_seen_pid   ON seen(person_id);
        ''')
        conn.commit()
        conn.close()

    def _migrate_v2(self):
        """Add department, employee_id, role columns if they don't exist yet."""
        conn = sqlite3.connect(CONFIG.db_path)
        cols = [r[1] for r in conn.execute('PRAGMA table_info(people)').fetchall()]
        for col, default in [('department', 'NULL'), ('employee_id', 'NULL'), ('role', "'employee'")]:
            if col not in cols:
                conn.execute(f'ALTER TABLE people ADD COLUMN {col} TEXT DEFAULT {default}')
        conn.commit()
        conn.close()

    def _migrate_v1(self):
        """Import any existing people folders that pre-date the people table."""
        if not os.path.exists(CONFIG.people_dir):
            return
        conn = sqlite3.connect(CONFIG.db_path)
        for folder in os.listdir(CONFIG.people_dir):
            if not os.path.isdir(os.path.join(CONFIG.people_dir, folder)):
                continue
            if not folder.isdigit():
                continue
            existing = conn.execute('SELECT id FROM people WHERE id = ?', (folder,)).fetchone()
            if existing:
                continue
            name, added = None, self._now()
            info_path = os.path.join(CONFIG.people_dir, folder, 'info.txt')
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('Name:'):
                            name = line.replace('Name:', '').strip() or None
                        elif line.startswith('First seen:'):
                            added = line.replace('First seen:', '').strip()
            conn.execute(
                'INSERT OR IGNORE INTO people (id, name, added) VALUES (?, ?, ?)',
                (folder, name, added)
            )
        conn.commit()
        conn.close()

    # ── Write operations ─────────────────────────────────────────────────────

    def add_person(self, pid: str):
        with self._lock:
            self._conn().execute(
                'INSERT OR IGNORE INTO people (id, added) VALUES (?, ?)',
                (pid, self._now())
            )
            self._conn().commit()

    def rename_person(self, pid: str, name: str):
        with self._lock:
            self._conn().execute('UPDATE people SET name = ? WHERE id = ?', (name, pid))
            self._conn().commit()

    def update_person_info(self, pid: str, name: str, department: str, employee_id: str, role: str):
        with self._lock:
            self._conn().execute(
                'UPDATE people SET name=?, department=?, employee_id=?, role=? WHERE id=?',
                (name or None, department or None, employee_id or None, role or 'employee', pid)
            )
            self._conn().commit()

    def delete_person(self, pid: str):
        with self._lock:
            c = self._conn()
            c.execute('DELETE FROM people  WHERE id        = ?', (pid,))
            c.execute('DELETE FROM visits  WHERE person_id = ?', (pid,))
            c.execute('DELETE FROM seen    WHERE person_id = ?', (pid,))
            c.commit()

    def log_visit(self, pid: str, action: str):
        with self._lock:
            self._conn().execute(
                'INSERT INTO visits (person_id, action, time) VALUES (?, ?, ?)',
                (pid, action, self._now())
            )
            self._conn().commit()

    def log_seen(self, pid: str):
        with self._lock:
            self._conn().execute(
                'INSERT INTO seen (person_id, time) VALUES (?, ?)',
                (pid, self._now())
            )
            self._conn().commit()

    # ── Read operations ──────────────────────────────────────────────────────

    def get_all_people(self) -> List[sqlite3.Row]:
        return self._conn().execute(
            '''
            SELECT p.id, p.name, p.department, p.employee_id, p.role, p.added,
                   COUNT(v.id)                                     AS visits,
                   MAX(v.time)                                     AS last_visit,
                   (SELECT MAX(s.time) FROM seen s
                    WHERE s.person_id = p.id)                     AS last_seen
            FROM people p
            LEFT JOIN visits v ON v.person_id = p.id
            GROUP BY p.id
            ORDER BY CAST(p.id AS INTEGER)
            '''
        ).fetchall()

    def get_person(self, pid: str) -> Optional[sqlite3.Row]:
        return self._conn().execute(
            'SELECT * FROM people WHERE id = ?', (pid,)
        ).fetchone()

    def get_visits(self, pid: str) -> List[sqlite3.Row]:
        return self._conn().execute(
            'SELECT id, action, time FROM visits WHERE person_id = ? ORDER BY id DESC',
            (pid,)
        ).fetchall()

    def get_seen_history(self, pid: str, limit: int = 100) -> List[sqlite3.Row]:
        return self._conn().execute(
            'SELECT id, time FROM seen WHERE person_id = ? ORDER BY id DESC LIMIT ?',
            (pid, limit)
        ).fetchall()

    def get_last_seen(self, pid: str) -> Optional[str]:
        row = self._conn().execute(
            'SELECT time FROM seen WHERE person_id = ? ORDER BY id DESC LIMIT 1', (pid,)
        ).fetchone()
        return row[0] if row else None

    def get_stats(self) -> dict:
        c = self._conn()
        return {
            'total_people': c.execute('SELECT COUNT(*) FROM people').fetchone()[0],
            'total_visits': c.execute('SELECT COUNT(*) FROM visits').fetchone()[0],
            'today_visits': c.execute(
                "SELECT COUNT(*) FROM visits WHERE date(time) = date('now')"
            ).fetchone()[0],
        }

    def get_today_counts(self) -> dict:
        c = self._conn()
        entries = c.execute(
            "SELECT COUNT(*) FROM visits WHERE action='ENTRY' AND date(time)=date('now')"
        ).fetchone()[0]
        exits = c.execute(
            "SELECT COUNT(*) FROM visits WHERE action='EXIT' AND date(time)=date('now')"
        ).fetchone()[0]
        total = c.execute('SELECT COUNT(*) FROM people').fetchone()[0]
        return {'entries': entries, 'exits': exits, 'total': total}

    def get_currently_inside(self) -> list:
        """People whose most recent event today was ENTRY."""
        return self._conn().execute(
            '''
            SELECT p.id, p.name, p.department, p.employee_id,
                   v.time AS entry_time
            FROM people p
            JOIN visits v ON v.person_id = p.id
            WHERE date(v.time) = date('now') AND v.action = 'ENTRY'
              AND v.id = (
                  SELECT MAX(v2.id) FROM visits v2
                  WHERE v2.person_id = p.id AND date(v2.time) = date('now')
              )
            ORDER BY v.time DESC
            '''
        ).fetchall()

    def get_recent_activity(self, limit: int = 15) -> list:
        """Latest entry/exit events across all people."""
        return self._conn().execute(
            '''
            SELECT p.name, p.department, v.action, v.time, v.person_id
            FROM visits v
            LEFT JOIN people p ON p.id = v.person_id
            ORDER BY v.id DESC LIMIT ?
            ''',
            (limit,)
        ).fetchall()

    def get_visits_by_day(self, pid: str, days: int = 7) -> List[sqlite3.Row]:
        return self._conn().execute(
            '''
            SELECT date(time) AS day, COUNT(*) AS cnt
            FROM visits
            WHERE person_id = ? AND time >= date('now', ?)
            GROUP BY day
            ORDER BY day
            ''',
            (pid, f'-{days} days')
        ).fetchall()


db = Database()
