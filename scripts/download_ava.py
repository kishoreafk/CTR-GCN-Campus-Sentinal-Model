"""
Download AVA dataset videos and annotations.
AVA provides annotations with ground-truth bounding boxes.
Videos sourced from YouTube via yt-dlp.
"""
import logging
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.db_manager import DBManager
from utils.class_registry import ClassRegistry

log = logging.getLogger("download_ava")

# AVA v2.2 annotation URLs (Google Cloud Storage)
ANNOTATION_URLS = {
    "ava_train_v2.2.csv":
        "https://research.google.com/ava/download/ava_train_v2.2.csv",
    "ava_val_v2.2.csv":
        "https://research.google.com/ava/download/ava_val_v2.2.csv",
    "ava_action_list_v2.2_for_activitynet_2019.pbtxt":
        "https://research.google.com/ava/download/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
}


def _download_annotations(data_dir: str):
    """Download AVA annotation files."""
    import requests
    ann_dir = Path(data_dir) / "annotations" / "ava"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for name, url in ANNOTATION_URLS.items():
        dest = ann_dir / name
        if dest.exists():
            log.info(f"Already present: {name}")
            continue
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            log.info(f"Downloaded: {name}")
        except Exception as e:
            log.error(f"Failed to download {name}: {e}")


def _download_video(video_id: str, output_dir: str,
                    db: DBManager, dataset: str) -> bool:
    """Download a single AVA video clip from YouTube."""
    if db.is_downloaded(video_id, dataset):
        return True

    db.mark_download_start(video_id, dataset)

    try:
        import subprocess
        output_path = Path(output_dir) / f"{video_id}.mp4"

        # AVA videos are 15 minutes each, from 902s to 1798s of original YouTube video
        result = subprocess.run([
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "-o", str(output_path),
            "-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
            "--merge-output-format", "mp4",
            "--download-sections", "*902-1798",
            "--no-playlist",
            "--quiet",
        ], capture_output=True, text=True, timeout=600)

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


def download_ava(config):
    """Main download function for AVA dataset."""
    log.info("Downloading AVA v2.2 dataset")

    # 1. Download annotations
    _download_annotations(config.data_dir)

    # 2. Parse annotations to get video IDs
    ann_dir = Path(config.data_dir) / "annotations" / "ava"
    video_ids = set()
    for csv_file in ann_dir.glob("*v2.2*.csv"):
        try:
            df = pd.read_csv(csv_file, header=None,
                             names=["video_id", "timestamp", "x1", "y1",
                                    "x2", "y2", "action_id", "person_id"])
            reg = ClassRegistry(config.class_config)
            df = reg.filter_annotations(df)
            video_ids.update(df["video_id"].unique())
        except Exception as e:
            log.warning(f"Could not parse {csv_file}: {e}")

    if not video_ids:
        log.warning("No video IDs found in annotations")
        return

    video_ids = list(video_ids)
    if config.test_mode:
        video_ids = video_ids[:config.test_max_videos]

    log.info(f"Found {len(video_ids)} unique video IDs to download")

    # 3. Download videos
    output_dir = Path(config.data_dir) / "raw" / "ava" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    db = DBManager(config.state_db)

    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_download_video, vid, str(output_dir),
                          db, "ava"): vid
            for vid in video_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Downloading AVA"):
            if future.result():
                success += 1
            else:
                failed += 1

    log.info(f"Download complete: {success} success, {failed} failed")
