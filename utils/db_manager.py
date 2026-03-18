"""
Thread-safe SQLite manager using WAL journal mode.
Each thread gets its own connection (threading.local).
All writes serialized via a single lock.
"""
import sqlite3, threading, logging
from pathlib import Path
from datetime import datetime, timedelta
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

    # ── Download Recovery API ────────────────────────────────────────────────

    def get_stale_downloads(self, dataset: str,
                            max_age_minutes: int = 30) -> List[str]:
        """
        Return video_ids stuck in 'downloading' for longer than max_age_minutes.
        These are downloads interrupted by crash/kill that never called mark_done.
        """
        cutoff = (datetime.utcnow() -
                  timedelta(minutes=max_age_minutes)).isoformat()
        with self.read() as c:
            c.execute("""
                SELECT video_id FROM downloads
                WHERE dataset=? AND status='downloading' AND updated_at < ?
            """, (dataset, cutoff))
            return [r["video_id"] for r in c.fetchall()]

    def reset_to_pending(self, video_id: str, dataset: str):
        """Reset a download back to pending for retry."""
        with self.write() as c:
            c.execute("""
                UPDATE downloads
                SET status='pending', last_error=NULL, updated_at=?
                WHERE video_id=? AND dataset=?
            """, (datetime.utcnow().isoformat(), video_id, dataset))

    def get_all_known_video_ids(self, dataset: str) -> set:
        """Return set of all video_ids tracked in downloads table."""
        with self.read() as c:
            c.execute(
                "SELECT video_id FROM downloads WHERE dataset=?", (dataset,)
            )
            return {r["video_id"] for r in c.fetchall()}

    def get_done_video_ids(self, dataset: str) -> List[tuple]:
        """Returns list of (video_id, file_path) for all done downloads."""
        with self.read() as c:
            c.execute("""
                SELECT video_id, file_path FROM downloads
                WHERE dataset=? AND status='done'
            """, (dataset,))
            return [(r["video_id"], r["file_path"]) for r in c.fetchall()]

    def get_attempt_count(self, video_id: str, dataset: str) -> int:
        """Return number of download attempts for a video."""
        with self.read() as c:
            c.execute("""
                SELECT attempts FROM downloads
                WHERE video_id=? AND dataset=?
            """, (video_id, dataset))
            row = c.fetchone()
            return row["attempts"] if row else 0

    def get_file_path(self, video_id: str, dataset: str) -> Optional[str]:
        """Return stored file_path for a video, or None."""
        with self.read() as c:
            c.execute("""
                SELECT file_path FROM downloads
                WHERE video_id=? AND dataset=?
            """, (video_id, dataset))
            row = c.fetchone()
            return row["file_path"] if row else None

    def retry_failed_downloads(self, dataset: str, max_attempts: int = 3):
        """
        Reset all 'failed' entries with fewer than max_attempts back to pending.
        Call at the start of a retry session.
        """
        with self.write() as c:
            c.execute("""
                UPDATE downloads
                SET status='pending', updated_at=?
                WHERE dataset=? AND status='failed' AND attempts < ?
            """, (datetime.utcnow().isoformat(), dataset, max_attempts))
        with self.read() as c:
            c.execute("""
                SELECT COUNT(*) cnt FROM downloads
                WHERE dataset=? AND status='pending'
            """, (dataset,))
            n = c.fetchone()["cnt"]
        log.info(f"Reset failed downloads → {n} pending for retry")

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

    # ── Annotation Recovery API ──────────────────────────────────────────────

    def mark_annotation_start(self, video_id: str, dataset: str,
                              timestamp_s: float, person_id: str):
        """Mark as 'annotating' so stale detection can find crashed jobs."""
        with self.write() as c:
            c.execute("""
                INSERT INTO annotations
                    (video_id, dataset, timestamp_s, person_id, status, updated_at)
                VALUES (?, ?, ?, ?, 'annotating', ?)
                ON CONFLICT DO UPDATE SET
                    status='annotating', updated_at=excluded.updated_at
            """, (video_id, dataset, timestamp_s, person_id,
                  datetime.utcnow().isoformat()))

    def get_stale_annotations(self, dataset: str,
                              max_age_minutes: int = 60) -> List[tuple]:
        """Return (video_id, timestamp_s, person_id) tuples stuck in 'annotating'."""
        cutoff = (datetime.utcnow() -
                  timedelta(minutes=max_age_minutes)).isoformat()
        with self.read() as c:
            c.execute("""
                SELECT video_id, timestamp_s, person_id
                FROM annotations
                WHERE dataset=? AND status='annotating' AND updated_at < ?
            """, (dataset, cutoff))
            return [(r["video_id"], r["timestamp_s"], r["person_id"])
                    for r in c.fetchall()]

    def reset_annotation_to_pending(self, video_id: str, dataset: str,
                                    timestamp_s: float, person_id: str):
        """Reset an annotation back to pending for retry."""
        with self.write() as c:
            c.execute("""
                UPDATE annotations
                SET status='pending', last_error=NULL, updated_at=?
                WHERE video_id=? AND dataset=? AND timestamp_s=? AND person_id=?
            """, (datetime.utcnow().isoformat(),
                  video_id, dataset, timestamp_s, person_id))

    def get_done_annotation_entries(self, dataset: str) -> List[tuple]:
        """Returns (video_id, timestamp_s, person_id, output_path) for done entries."""
        with self.read() as c:
            c.execute("""
                SELECT video_id, timestamp_s, person_id, output_path
                FROM annotations
                WHERE dataset=? AND status='done'
            """, (dataset,))
            return [(r["video_id"], r["timestamp_s"],
                     r["person_id"], r["output_path"])
                    for r in c.fetchall()]

    # ── DB Backup ────────────────────────────────────────────────────────────

    def backup(self, backup_dir: str = None) -> Path:
        """
        Create a timestamped backup of the SQLite DB.
        Uses SQLite's online backup API — safe even while DB is open.
        Keeps only last 5 backups.
        """
        backup_dir = Path(backup_dir or self.db_path.parent / "backups")
        backup_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = backup_dir / f"state_backup_{ts}.db"

        src_conn = sqlite3.connect(str(self.db_path))
        dest_conn = sqlite3.connect(str(dest))
        src_conn.backup(dest_conn)
        dest_conn.close()
        src_conn.close()

        log.info(f"DB backed up to: {dest}")

        # Keep only last 5 backups
        old_backups = sorted(backup_dir.glob("state_backup_*.db"))
        for old in old_backups[:-5]:
            old.unlink()

        return dest
