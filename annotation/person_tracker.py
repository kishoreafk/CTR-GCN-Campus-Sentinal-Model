"""
Lightweight IoU-based person tracker.
Ensures consistent identity across the 64 frames of a clip.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Track:
    track_id: int
    bbox: np.ndarray       # (4,) [x1,y1,x2,y2]
    keypoints: np.ndarray  # (17, 2)
    scores: np.ndarray     # (17,)
    age: int = 0
    hits: int = 1

def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

class PersonTracker:
    def __init__(self, max_persons: int = 2,
                 iou_thr: float = 0.4, max_age: int = 5):
        self.max_persons = max_persons
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(self, detections: List[dict]) -> List[Track]:
        """
        detections: list of {bbox:(4,), keypoints:(17,2), scores:(17,)}
        Returns active tracks in stable track_id order.
        """
        if not self.tracks:
            for det in detections[:self.max_persons]:
                self.tracks.append(
                    Track(self._next_id, det["bbox"],
                          det["keypoints"], det["scores"]))
                self._next_id += 1
            return list(self.tracks)

        n_t, n_d = len(self.tracks), len(detections)
        if n_d == 0:
            survived = [Track(t.track_id, t.bbox, t.keypoints, t.scores,
                              t.age+1, t.hits)
                        for t in self.tracks if t.age+1 <= self.max_age]
            self.tracks = survived
            return list(self.tracks)

        cost = np.array([[iou(t.bbox, d["bbox"])
                          for d in detections]
                         for t in self.tracks])

        matched_t, matched_d = set(), set()
        matches = []
        for idx in np.argsort(cost.ravel())[::-1]:
            ti, di = divmod(idx, n_d)
            if cost[ti, di] < self.iou_thr:
                break
            if ti in matched_t or di in matched_d:
                continue
            matches.append((ti, di))
            matched_t.add(ti); matched_d.add(di)

        new_tracks = []
        for ti, di in matches:
            t = self.tracks[ti]; d = detections[di]
            new_tracks.append(Track(t.track_id, d["bbox"],
                                    d["keypoints"], d["scores"],
                                    0, t.hits+1))

        for ti, t in enumerate(self.tracks):
            if ti not in matched_t and t.age+1 <= self.max_age:
                new_tracks.append(Track(t.track_id, t.bbox, t.keypoints,
                                        t.scores, t.age+1, t.hits))

        for di, d in enumerate(detections):
            if di not in matched_d and len(new_tracks) < self.max_persons:
                new_tracks.append(Track(self._next_id, d["bbox"],
                                        d["keypoints"], d["scores"]))
                self._next_id += 1

        new_tracks.sort(key=lambda t: t.track_id)
        self.tracks = new_tracks[:self.max_persons]
        return list(self.tracks)

    def reset(self):
        self.tracks = []
        self._next_id = 0
