"""
Thread-safe SQLite manager using WAL journal mode.
Each thread gets its own connection (threading.local).
All writes serialized via a single lock.
"""
import sqlite3, threading, logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Optional

log = logging.getLogger("db_manager")


class DBManager:
    _local = threading.local()

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(str(self.db_path),
                                   check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def read(self):
        yield self._connect().cursor()

    @contextmanager
    def write(self):
        with self._write_lock:
            conn = self._connect()
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _init_schema(self):
        with self.write() as cur:
            cur.executescript("""
            CREATE TABLE IF NOT EXISTS downloads (
                video_id    TEXT NOT NULL,
                dataset     TEXT NOT NULL,
                status      TEXT DEFAULT 'pending',
                source      TEXT,
                attempts    INTEGER DEFAULT 0,
                file_path   TEXT,
                duration_s  REAL,
                last_error  TEXT,
                updated_at  TEXT,
                PRIMARY KEY (video_id, dataset)
            );
            CREATE TABLE IF NOT EXISTS annotations (
                video_id      TEXT NOT NULL,
                dataset       TEXT NOT NULL,
                timestamp_s   REAL NOT NULL,
                person_id     TEXT NOT NULL,
                status        TEXT DEFAULT 'pending',
                output_path   TEXT,
                quality_score REAL,
                last_error    TEXT,
                updated_at    TEXT,
                PRIMARY KEY (video_id, dataset, timestamp_s, person_id)
            );
            """)

    # ── Download API ─────────────────────────────────────────────────────────

    def is_downloaded(self, video_id: str, dataset: str) -> bool:
        with self.read() as c:
            c.execute("SELECT status FROM downloads WHERE video_id=? AND dataset=?",
                      (video_id, dataset))
            row = c.fetchone()
            return bool(row and row["status"] == "done")

    def mark_download_start(self, video_id: str, dataset: str):
        ts = datetime.utcnow().isoformat()
        with self.write() as c:
            c.execute("""
                INSERT INTO downloads (video_id,dataset,status,attempts,updated_at)
                VALUES (?,?,'downloading',1,?)
                ON CONFLICT(video_id,dataset) DO UPDATE SET
                    status='downloading', attempts=attempts+1, updated_at=excluded.updated_at
            """, (video_id, dataset, ts))

    def mark_download_done(self, video_id: str, dataset: str,
                           source: str, file_path: str, duration_s: float):
        with self.write() as c:
            c.execute("""
                UPDATE downloads SET status='done',source=?,file_path=?,
                    duration_s=?,last_error=NULL,updated_at=?
                WHERE video_id=? AND dataset=?
            """, (source, file_path, duration_s,
                  datetime.utcnow().isoformat(), video_id, dataset))

    def mark_download_failed(self, video_id: str, dataset: str, error: str):
        with self.write() as c:
            c.execute("""
                UPDATE downloads SET status='failed',last_error=?,updated_at=?
                WHERE video_id=? AND dataset=?
            """, (error, datetime.utcnow().isoformat(), video_id, dataset))

    def get_pending_downloads(self, dataset: str,
                               max_attempts: int = 3) -> List[Dict]:
        with self.read() as c:
            c.execute("""
                SELECT video_id, attempts FROM downloads
                WHERE dataset=? AND status!='done' AND attempts<?
                ORDER BY attempts ASC
            """, (dataset, max_attempts))
            return [dict(r) for r in c.fetchall()]

    # ── Annotation API ───────────────────────────────────────────────────────

    def is_annotated(self, video_id: str, dataset: str,
                     timestamp_s: float, person_id: str) -> bool:
        with self.read() as c:
            c.execute("""
                SELECT status FROM annotations
                WHERE video_id=? AND dataset=? AND timestamp_s=? AND person_id=?
            """, (video_id, dataset, timestamp_s, person_id))
            row = c.fetchone()
            return bool(row and row["status"] == "done")

    def mark_annotation_done(self, video_id: str, dataset: str,
                              timestamp_s: float, person_id: str,
                              output_path: str, quality_score: float):
        with self.write() as c:
            c.execute("""
                INSERT INTO annotations
                    (video_id,dataset,timestamp_s,person_id,status,
                     output_path,quality_score,updated_at)
                VALUES (?,?,?,?,'done',?,?,?)
                ON CONFLICT DO UPDATE SET
                    status='done', output_path=excluded.output_path,
                    quality_score=excluded.quality_score,
                    last_error=NULL, updated_at=excluded.updated_at
            """, (video_id, dataset, timestamp_s, person_id,
                  output_path, quality_score, datetime.utcnow().isoformat()))

    def mark_annotation_failed(self, video_id: str, dataset: str,
                                timestamp_s: float, person_id: str, error: str):
        with self.write() as c:
            c.execute("""
                INSERT INTO annotations
                    (video_id,dataset,timestamp_s,person_id,status,last_error,updated_at)
                VALUES (?,?,?,?,'failed',?,?)
                ON CONFLICT DO UPDATE SET
                    status='failed', last_error=excluded.last_error,
                    updated_at=excluded.updated_at
            """, (video_id, dataset, timestamp_s, person_id,
                  error, datetime.utcnow().isoformat()))

    def annotation_summary(self, dataset: str) -> Dict[str, int]:
        with self.read() as c:
            c.execute("""
                SELECT status, COUNT(*) cnt FROM annotations
                WHERE dataset=? GROUP BY status
            """, (dataset,))
            return {r["status"]: r["cnt"] for r in c.fetchall()}
