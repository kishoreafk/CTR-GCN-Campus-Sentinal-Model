"""
Download AVA-Kinetics videos and annotations.
Uses yt-dlp for YouTube video downloads.
State tracked in SQLite DB for idempotent re-runs.
"""
import logging
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.db_manager import DBManager
from utils.class_registry import ClassRegistry

log = logging.getLogger("download_ava_kinetics")

# AVA-Kinetics annotation URLs (Google Cloud Storage)
ANNOTATION_URLS = {
    "ava_kinetics_train_v1.0.csv":
        "https://storage.googleapis.com/deepmind-media/Datasets/ava_kinetics_v1_0/ava_kinetics_train_v1.0.csv",
    "ava_kinetics_val_v1.0.csv":
        "https://storage.googleapis.com/deepmind-media/Datasets/ava_kinetics_v1_0/ava_kinetics_val_v1.0.csv",
}


def download_with_backoff(url: str, dest: Path, max_retries: int = 5, initial_wait: int = 2, max_wait: int = 60) -> bool:
    import requests, time
    wait = initial_wait
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code in (404, 403, 401):
                log.error(f"Fatal HTTP {r.status_code} for {url} - will not retry")
                return False
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except requests.RequestException as e:
            log.warning(f"Download failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                log.info(f"Retrying in {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, max_wait)
    log.error(f"Failed to download {url} after {max_retries} attempts.")
    return False


def _download_annotations(data_dir: str):
    """Download AVA-Kinetics annotation CSVs."""
    ann_dir = Path(data_dir) / "annotations" / "ava_kinetics"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for name, url in ANNOTATION_URLS.items():
        dest = ann_dir / name
        if dest.exists():
            log.info(f"Annotation already present: {name}")
            continue
        if download_with_backoff(url, dest):
            log.info(f"Downloaded annotation: {name}")
        else:
            log.error(f"Failed to download {name}")


def _download_video(video_id: str, output_dir: str,
                    db: DBManager, dataset: str) -> bool:
    """Download a single YouTube video clip using yt-dlp."""
    if db.is_downloaded(video_id, dataset):
        return True

    db.mark_download_start(video_id, dataset)

    try:
        import subprocess
        output_path = Path(output_dir) / f"{video_id}.mp4"
        result = subprocess.run([
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "-o", str(output_path),
            "-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--quiet",
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and output_path.exists():
            import cv2
            cap = cv2.VideoCapture(str(output_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total / fps if fps > 0 else 0
            cap.release()

            db.mark_download_done(video_id, dataset, "youtube",
                                  str(output_path), duration)
            return True
        else:
            db.mark_download_failed(video_id, dataset,
                                    result.stderr[:500] if result.stderr else "Unknown error")
            return False

    except Exception as e:
        db.mark_download_failed(video_id, dataset, str(e))
        return False


def download_ava_kinetics(config, selected_registry=None):
    """
    Main download function for AVA-Kinetics dataset.

    Parameters
    ----------
    config             : TrainingConfig
    selected_registry  : ClassRegistry subset (only selected classes).
                         If None, uses all classes from config.
    """
    log.info("Downloading AVA-Kinetics dataset")

    if selected_registry is None:
        selected_registry = ClassRegistry(config.class_config)

    # 1. Download annotations
    _download_annotations(config.data_dir)

    # 2. Parse annotations to get video IDs — filtered to selected classes
    ann_dir = Path(config.data_dir) / "annotations" / "ava_kinetics"
    video_ids = set()
    for csv_file in ann_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, header=None,
                             names=["video_id", "timestamp", "x1", "y1",
                                    "x2", "y2", "action_id", "person_id"])
            df = selected_registry.filter_annotations(df)
            video_ids.update(df["video_id"].unique())
        except Exception as e:
            log.warning(f"Could not parse {csv_file}: {e}")

    if not video_ids:
        log.warning("No video IDs found in annotations for selected classes")
        return

    # In test mode, limit downloads
    video_ids = list(video_ids)
    if config.test_mode:
        video_ids = video_ids[:config.test_max_videos]

    log.info(f"Classes selected: {selected_registry.class_names}")
    log.info(f"Found {len(video_ids)} unique video IDs to download")

    # 3. Download videos
    output_dir = Path(config.data_dir) / "raw" / "ava_kinetics" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from utils.system_checks import check_disk_before_download
    if not check_disk_before_download(video_ids, str(output_dir), avg_video_mb=150.0):
        log.warning("Skipping download due to low disk space.")
        return

    db = DBManager(config.state_db)

    # Register all videos in DB (skips already-done ones)
    for vid in video_ids:
        if not db.is_downloaded(vid, "ava_kinetics"):
            # Check if file already exists on disk
            existing = _find_existing_video(output_dir, vid)
            if existing:
                log.info(f"Found existing file for {vid} — marking done")
                db.mark_download_done(vid, "ava_kinetics", "existing_file",
                                      str(existing), 0.0)
            else:
                db.mark_download_start(vid, "ava_kinetics")

    # Download with thread pool
    success, failed, skipped = 0, 0, 0
    with ThreadPoolExecutor(max_workers=getattr(config, 'download_workers', 4)) as executor:
        futures = {
            executor.submit(_download_video, vid, str(output_dir),
                          db, "ava_kinetics"): vid
            for vid in video_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Downloading"):
            result = future.result()
            if result == "skipped":
                skipped += 1
            elif result:
                success += 1
            else:
                failed += 1

    log.info(f"Download complete: {success} new, {skipped} skipped, {failed} failed")


def _find_existing_video(video_dir: Path, video_id: str):
    """Check if a video file already exists on disk."""
    for ext in (".mp4", ".avi", ".mkv", ".webm"):
        p = video_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    # Also try files that start with video_id (yt-dlp may add suffix)
    matches = list(video_dir.glob(f"{video_id}*"))
    return matches[0] if matches else None

