"""Tests for thread-safe SQLite DB manager."""
import pytest
import threading
import tempfile
from pathlib import Path
from utils.db_manager import DBManager


@pytest.fixture
def db(tmp_path):
    return DBManager(str(tmp_path / "test.db"))


def test_concurrent_writes(db):
    """100 threads writing simultaneously should not corrupt the DB."""
    errors = []

    def worker(vid):
        try:
            db.mark_download_start(f"video_{vid}", "test")
            db.mark_download_done(f"video_{vid}", "test", "youtube",
                                  f"/tmp/video_{vid}.mp4", 10.0)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(errors) == 0, f"Errors: {errors}"


def test_idempotent_done(db):
    """Marking done twice should be safe."""
    db.mark_download_start("v1", "ava")
    db.mark_download_done("v1", "ava", "youtube", "/tmp/v1.mp4", 10.0)
    # Second call should not raise
    db.mark_download_start("v1", "ava")
    db.mark_download_done("v1", "ava", "youtube", "/tmp/v1.mp4", 10.0)
    assert db.is_downloaded("v1", "ava")


def test_pending_filters_done(db):
    """Done videos should not appear in pending list."""
    db.mark_download_start("v1", "ava")
    db.mark_download_done("v1", "ava", "youtube", "/tmp/v1.mp4", 10.0)
    db.mark_download_start("v2", "ava")

    pending = db.get_pending_downloads("ava")
    video_ids = [p["video_id"] for p in pending]
    assert "v1" not in video_ids
    assert "v2" in video_ids


def test_annotation_key(db):
    """4-part PK (video, dataset, ts, person) should work."""
    db.mark_annotation_done("v1", "ava", 5.0, "p1", "/out/1.npz", 0.9)
    db.mark_annotation_done("v1", "ava", 5.0, "p2", "/out/2.npz", 0.8)
    db.mark_annotation_done("v1", "ava", 6.0, "p1", "/out/3.npz", 0.7)

    assert db.is_annotated("v1", "ava", 5.0, "p1")
    assert db.is_annotated("v1", "ava", 5.0, "p2")
    assert db.is_annotated("v1", "ava", 6.0, "p1")
    assert not db.is_annotated("v1", "ava", 7.0, "p1")
