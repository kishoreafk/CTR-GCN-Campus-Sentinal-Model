"""
Prints a summary table of what has been downloaded and annotated
for each selected class.

Invoked by: python main.py --mode status --classes eat drink walk
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

log = logging.getLogger("pipeline_status")


def print_pipeline_status(config, selected_registry):
    """
    Prints a table like:

    Class               DL Done  DL Fail  Ann Done  Ann Fail
    ─────────────────────────────────────────────────────────
    eat                     142        3       138         4
    drink                    97        1        97         0
    walk                    203        5       198         5
    ─────────────────────────────────────────────────────────
    TOTAL                   442        9       433         9

    Train skeletons : 387
    Val skeletons   : 46
    Ready to train  : YES
    """
    from utils.db_manager import DBManager
    db = DBManager(config.state_db)

    ann_csv = _load_annotation_df(config)
    skel_dir = Path(config.data_dir) / "processed" / config.dataset / "skeletons"

    rows = []
    total_dl_done = total_dl_fail = total_ann_done = total_ann_fail = 0

    for cls in selected_registry._classes:
        cls_id = cls["id"]
        cls_name = cls["name"]

        # Count download stats
        video_ids = _get_video_ids_for_class(ann_csv, cls_id)
        dl_done = sum(1 for v in video_ids
                      if db.is_downloaded(v, config.dataset))
        dl_fail = len(video_ids) - dl_done

        # Count annotation stats from DB
        ann_done = _count_annotations(db, config.dataset, cls_id, "done")
        ann_fail = _count_annotations(db, config.dataset, cls_id, "failed")

        rows.append((cls_name, dl_done, dl_fail, ann_done, ann_fail))
        total_dl_done += dl_done
        total_dl_fail += dl_fail
        total_ann_done += ann_done
        total_ann_fail += ann_fail

    # Count skeleton files on disk
    skel_count = len(list(skel_dir.rglob("*.npz"))) if skel_dir.exists() else 0

    # Print table
    w = max((len(r[0]) for r in rows), default=10) + 2
    header = (f"{'Class':<{w}} {'DL Done':>8} {'DL Fail':>8} "
              f"{'Ann Done':>9} {'Ann Fail':>9}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for cls_name, dl_done, dl_fail, ann_done, ann_fail in rows:
        print(f"{cls_name:<{w}} {dl_done:>8} {dl_fail:>8} "
              f"{ann_done:>9} {ann_fail:>9}")
    print(sep)
    print(f"{'TOTAL':<{w}} {total_dl_done:>8} {total_dl_fail:>8} "
          f"{total_ann_done:>9} {total_ann_fail:>9}")
    print(sep)
    print(f"\nSkeleton files on disk : {skel_count}")
    print(f"Ready to train         : {'YES' if total_ann_done > 0 else 'NO'}\n")


def _load_annotation_df(config) -> Optional[pd.DataFrame]:
    """Load annotation CSV for the configured dataset."""
    ann_dir = Path(config.data_dir) / "annotations" / config.dataset
    for pattern in ["*.csv"]:
        csvs = list(ann_dir.glob(pattern))
        if csvs:
            frames = []
            for csv_path in csvs:
                try:
                    df = pd.read_csv(csv_path, header=None,
                                     names=["video_id", "timestamp", "x1", "y1",
                                            "x2", "y2", "action_id", "person_id"])
                    frames.append(df)
                except Exception:
                    continue
            if frames:
                return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _get_video_ids_for_class(ann_df: pd.DataFrame, class_id: int) -> list:
    """Get unique video IDs that contain annotations for a specific class."""
    if ann_df.empty:
        return []
    filtered = ann_df[ann_df["action_id"] == class_id]
    return filtered["video_id"].unique().tolist()


def _count_annotations(db, dataset: str, class_id: int,
                       status: str = "done") -> int:
    """
    Count annotations for a specific class from the DB.
    This is approximate — we count based on annotation rows that
    have the matching action_id stored in their output path or metadata.
    For now, returns total count by status (not per-class).
    """
    # Since the DB doesn't store per-class info directly, we return
    # an aggregate count. A future improvement could add action_id to
    # the annotations table.
    try:
        with db.read() as c:
            c.execute("""
                SELECT COUNT(*) cnt FROM annotations
                WHERE dataset=? AND status=?
            """, (dataset, status))
            row = c.fetchone()
            return row["cnt"] if row else 0
    except Exception:
        return 0
