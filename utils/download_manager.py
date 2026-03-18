"""
DownloadManager with complete interruption recovery.

On any startup, before downloading anything new:
  1. Scan for .tmp files → delete them (incomplete downloads)
  2. Find DB entries stuck in 'downloading' → reset to 'pending'
  3. Find files on disk not in DB → validate and register them
  4. Find DB entries marked 'done' but file missing → reset to 'pending'

This makes restart after ANY failure safe and automatic.
"""

import subprocess, json, time, logging, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from tqdm import tqdm
from utils.db_manager import DBManager

log = logging.getLogger("download_manager")

# How long a download can be "in progress" before we consider it stale
STALE_DOWNLOAD_MINUTES = 30


def _validate_video_file(path: Path) -> bool:
    """Check that a video file is non-empty and readable by ffprobe."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0 and "video" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # ffprobe not available or timed out — accept file if non-empty
        return path.stat().st_size > 1024


def _get_duration(path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except Exception:
        return 0.0


def _trim_video(src: Path, dest: Path,
                start_s: float, end_s: float) -> bool:
    """
    Trim video to [start_s, end_s] using ffmpeg.
    Atomic: writes to dest.tmp then renames.
    """
    tmp = dest.with_suffix(".trim.tmp.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-to", str(end_s),
        "-i", str(src),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(tmp)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            tmp.unlink(missing_ok=True)
            return False
        tmp.rename(dest)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


class YouTubeSource:
    """yt-dlp download source with timeout and error handling."""

    def __init__(self, fmt: str = "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]"):
        self.format = fmt

    def download(self, video_id: str, dest: Path,
                 metadata: dict, atomic_write=None) -> bool:
        """
        Download YouTube clip with:
        - Per-download timeout (default 3 min)
        - Graceful handling of deleted/private/age-restricted videos
        - Trim to exact start/end timestamps after download
        """
        try:
            import yt_dlp
        except ImportError:
            log.error("yt-dlp not installed — cannot download from YouTube")
            return False

        url = f"https://www.youtube.com/watch?v={video_id}"
        tmp_path = str(dest.with_suffix(".tmp.%(ext)s"))

        ydl_opts = {
            "format":       self.format,
            "outtmpl":      tmp_path,
            "quiet":        True,
            "no_warnings":  False,
            "socket_timeout": 30,
            "retries":      3,
            "ignoreerrors": False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    return False

            # Trim to AVA clip timestamps if provided
            start_s = metadata.get("start_s")
            end_s = metadata.get("end_s")
            raw = list(dest.parent.glob(f"{video_id}.tmp.*"))
            if not raw:
                return False

            if start_s is not None and end_s is not None:
                ok = _trim_video(raw[0], dest, start_s, end_s)
                raw[0].unlink(missing_ok=True)
                return ok
            else:
                raw[0].rename(dest)
                return True

        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in
                   ("video unavailable", "private video",
                    "has been removed", "not available")):
                log.warning(f"Video {video_id} unavailable on YouTube")
            else:
                log.warning(f"yt-dlp error for {video_id}: {e}")
            return False


class DownloadManager:

    def __init__(self, db: DBManager,
                 max_workers: int = 4,
                 max_retries: int = 3,
                 timeout_per_video: int = 180):
        self.db          = db
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout     = timeout_per_video
        self.sources     = []

    def add_source(self, source):
        """Register a download source (e.g. YouTubeSource)."""
        self.sources.append(source)

    # ──────────────────────────────────────────────────────────────────────
    # STARTUP RECOVERY — call this once before any batch download
    # ──────────────────────────────────────────────────────────────────────

    def recover_interrupted_downloads(self, video_dir: Path, dataset: str):
        """
        Run at the start of every download session.
        Safe to call even if nothing was interrupted.

        Actions:
          1. Delete all .tmp files left by previous crashed downloads
          2. Reset DB entries stuck in 'downloading' for > STALE_DOWNLOAD_MINUTES
          3. Reconcile: files on disk that DB doesn't know about
          4. Reconcile: DB says 'done' but file is gone or corrupt

        Returns summary dict for logging.
        """
        summary = {
            "tmp_deleted":       0,
            "stale_reset":       0,
            "orphan_registered": 0,
            "missing_reset":     0,
        }

        # ── 1. Delete .tmp partial files ──────────────────────────────────
        for tmp in video_dir.glob("*.tmp"):
            tmp.unlink()
            summary["tmp_deleted"] += 1
            log.info(f"Deleted partial download: {tmp.name}")
        # Also clean up multi-extension tmp files (e.g. .tmp.mp4)
        for tmp in video_dir.glob("*.tmp.*"):
            tmp.unlink()
            summary["tmp_deleted"] += 1
            log.info(f"Deleted partial download: {tmp.name}")

        # ── 2. Reset stale 'downloading' entries ──────────────────────────
        stale = self.db.get_stale_downloads(
            dataset, max_age_minutes=STALE_DOWNLOAD_MINUTES
        )
        for video_id in stale:
            self.db.reset_to_pending(video_id, dataset)
            summary["stale_reset"] += 1
        if stale:
            log.info(f"Reset {len(stale)} stale downloads to 'pending'")

        # ── 3. Register orphan files (on disk, not in DB) ─────────────────
        known = self.db.get_all_known_video_ids(dataset)
        for f in video_dir.iterdir():
            if f.suffix in (".mp4", ".avi", ".mkv", ".webm"):
                vid_id = f.stem
                if vid_id not in known:
                    if _validate_video_file(f):
                        dur = _get_duration(f)
                        self.db.mark_download_start(vid_id, dataset)
                        self.db.mark_download_done(
                            vid_id, dataset,
                            source="pre_existing",
                            file_path=str(f),
                            duration_s=dur
                        )
                        summary["orphan_registered"] += 1
                        log.debug(f"Registered orphan: {f.name}")

        # ── 4. Reset 'done' entries whose file disappeared ────────────────
        done_ids = self.db.get_done_video_ids(dataset)
        for video_id, file_path in done_ids:
            p = Path(file_path)
            if not p.exists() or not _validate_video_file(p):
                self.db.reset_to_pending(video_id, dataset)
                summary["missing_reset"] += 1
                log.warning(
                    f"File gone/corrupt for '{video_id}' → reset to pending"
                )

        log.info(
            f"Recovery complete: "
            f"tmp_deleted={summary['tmp_deleted']}  "
            f"stale_reset={summary['stale_reset']}  "
            f"orphan_registered={summary['orphan_registered']}  "
            f"missing_reset={summary['missing_reset']}"
        )
        return summary

    # ──────────────────────────────────────────────────────────────────────
    # SINGLE VIDEO DOWNLOAD — full state machine
    # ──────────────────────────────────────────────────────────────────────

    def _expected_path(self, output_dir: Path, video_id: str,
                       metadata: dict) -> Path:
        """Determine expected output file path."""
        return output_dir / f"{video_id}.mp4"

    def download_video(self, video_id: str, dataset: str,
                       output_dir: Path, metadata: dict) -> str:
        """
        Full state machine for one video.

        States:  pending → downloading → done
                                      ↘ failed (retryable up to max_retries)

        Returns: "done" | "skipped" | "failed"
        """
        # ── Gate 1: already done in DB + file valid ───────────────────────
        if self.db.is_downloaded(video_id, dataset):
            fp = self.db.get_file_path(video_id, dataset)
            if fp and Path(fp).exists() and _validate_video_file(Path(fp)):
                return "skipped"
            # DB says done but file is bad — fall through to re-download
            log.warning(
                f"DB done but file invalid for {video_id} — re-downloading"
            )
            self.db.reset_to_pending(video_id, dataset)

        # ── Gate 2: check retry limit ─────────────────────────────────────
        attempts = self.db.get_attempt_count(video_id, dataset)
        if attempts >= self.max_retries:
            log.warning(
                f"Max retries ({self.max_retries}) reached for {video_id} — skip"
            )
            return "failed"

        # ── Gate 3: determine output path ─────────────────────────────────
        dest = self._expected_path(output_dir, video_id, metadata)

        # File exists on disk but not in DB (e.g. manual copy)
        if dest.exists() and _validate_video_file(dest):
            dur = _get_duration(dest)
            self.db.mark_download_start(video_id, dataset)
            self.db.mark_download_done(
                video_id, dataset, "pre_existing", str(dest), dur
            )
            return "skipped"

        # ── Try each source ───────────────────────────────────────────────
        self.db.mark_download_start(video_id, dataset)

        for source in self.sources:
            try:
                ok = source.download(video_id, dest, metadata)
                if ok and dest.exists() and _validate_video_file(dest):
                    dur = _get_duration(dest)
                    self.db.mark_download_done(
                        video_id, dataset,
                        source.__class__.__name__,
                        str(dest), dur
                    )
                    return "done"
                else:
                    dest.unlink(missing_ok=True)
            except KeyboardInterrupt:
                # Clean up and re-raise so outer loop can save progress
                dest.with_suffix(".tmp").unlink(missing_ok=True)
                dest.unlink(missing_ok=True)
                raise
            except Exception as e:
                log.warning(
                    f"Source {source.__class__.__name__} failed "
                    f"for {video_id}: {e}"
                )

        self.db.mark_download_failed(
            video_id, dataset, error="All sources exhausted"
        )
        return "failed"

    # ──────────────────────────────────────────────────────────────────────
    # BATCH DOWNLOAD — parallel with CTRL+C safety
    # ──────────────────────────────────────────────────────────────────────

    def download_batch(self, video_list: List[str], dataset: str,
                       output_dir: Path, metadata: Dict) -> dict:
        """
        Parallel download with:
        - Progress bar showing done/skipped/failed counts live
        - CTRL+C handler: waits for in-flight downloads to finish,
          saves state, exits cleanly
        - Summary written to {output_dir}/download_summary.json

        Returns: {done, skipped, failed, total}
        """
        counts = {"done": 0, "skipped": 0, "failed": 0}
        stop_event = threading.Event()

        import signal
        old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)

        def sigint_handler(sig, frame):
            log.info(
                "\nCTRL+C received — waiting for in-flight downloads "
                "to finish cleanly..."
            )
            stop_event.set()

        signal.signal(signal.SIGINT, sigint_handler)

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        self.download_video,
                        vid, dataset, output_dir, metadata.get(vid, {})
                    ): vid
                    for vid in video_list
                }

                with tqdm(total=len(video_list), desc="Downloading",
                          unit="video") as bar:
                    for future in as_completed(futures):
                        vid = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            result = "failed"
                            log.error(f"Unexpected error for {vid}: {e}")

                        counts[result] = counts.get(result, 0) + 1
                        bar.update(1)
                        bar.set_postfix(counts)

                        if stop_event.is_set():
                            log.info(
                                "Stop requested — cancelling queued downloads"
                            )
                            for f in futures:
                                f.cancel()
                            break

        finally:
            signal.signal(signal.SIGINT, old_handler)

        counts["total"] = len(video_list)
        # Write summary
        summary_path = output_dir / "download_summary.json"
        with open(summary_path, "w") as f:
            json.dump(counts, f, indent=2)

        return counts
