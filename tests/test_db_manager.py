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


def test_download_recovery_methods(db):
    """Test get_stale_downloads, reset_to_pending, etc."""
    # Insert a stale record
    with db.write() as cur:
        cur.execute('''
            INSERT INTO downloads (video_id, dataset, status, updated_at)
            VALUES (?, ?, ?, datetime('now', '-3 hours'))
        ''', ("stale_vid", "ava", "downloading"))
    
    # Should find the stale download
    stale = db.get_stale_downloads("ava", max_age_minutes=120)
    assert "stale_vid" in stale
    
    # Reset it
    db.reset_to_pending("stale_vid", "ava")
    assert db.get_attempt_count("stale_vid", "ava") == 0
    
    # Now it shouldn't be stale
    assert "stale_vid" not in db.get_stale_downloads("ava", max_age_minutes=120)


def test_annotation_recovery_methods(db):
    """Test get_stale_annotations, reset_annotation_to_pending, etc."""
    with db.write() as cur:
        cur.execute('''
            INSERT INTO annotations 
            (video_id, dataset, timestamp_s, person_id, status, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now', '-3 hours'))
        ''', ("v_ann", "ava", 1.0, "p1", "annotating"))
    
    stale = db.get_stale_annotations("ava", max_age_minutes=120)
    assert len(stale) == 1
    assert stale[0][0] == "v_ann"
    
    db.reset_annotation_to_pending("v_ann", "ava", 1.0, "p1")
    
    stale_after = db.get_stale_annotations("ava", max_age_minutes=120)
    assert len(stale_after) == 0
