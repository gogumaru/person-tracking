import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from boxmot.reid.core.reid import ReID

from config import (
    YOLO_MODEL, REID_WEIGHTS, TRACKER,
    CONF_THRESHOLD, REID_INTERVAL, REID_SIMILARITY_THRESH, REID_MAX_GALLERY,
)


class PersonTracker:
    """
    YOLO + BoT-SORT + OSNet Re-ID.
    Setiap call update() menerima frame, mengembalikan list track result.
    """

    def __init__(self):
        self.model = YOLO(str(YOLO_MODEL))

        self.reid = ReID(
            path=Path(REID_WEIGHTS),
            device=torch.device("mps"),
            half=False,
        )

        # gallery: person_id → [embeddings]
        self.gallery: dict[int, list] = {}
        self.next_pid   = 1
        self.track_to_pid: dict[int, int] = {}
        self.frame_count = 0

    # ── Re-ID helpers ──────────────────────────────────────────────────────────

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def _match_or_create(self, emb: np.ndarray) -> int:
        best_id, best_sim = None, -1.0
        for pid, embs in self.gallery.items():
            sim = self._cosine_sim(emb, np.mean(embs, axis=0))
            if sim > best_sim:
                best_sim, best_id = sim, pid

        if best_id is not None and best_sim >= REID_SIMILARITY_THRESH:
            self.gallery[best_id].append(emb)
            if len(self.gallery[best_id]) > REID_MAX_GALLERY:
                self.gallery[best_id].pop(0)
            return best_id

        pid = self.next_pid
        self.next_pid += 1
        self.gallery[pid] = [emb]
        return pid

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[dict]:
        """
        Proses satu frame.

        Returns:
            list of dict:
              {
                "person_id": int,   # persistent ID (via Re-ID)
                "track_id":  int,   # ID dari BoT-SORT (bisa berubah saat re-entry)
                "bbox":      [x1, y1, x2, y2],
                "conf":      float,
              }
        """
        self.frame_count += 1

        results = self.model.track(
            source=frame,
            conf=CONF_THRESHOLD,
            classes=0,
            tracker=TRACKER,
            persist=True,
            verbose=False,
        )

        if results[0].boxes is None:
            return []

        valid = [b for b in results[0].boxes if b.id is not None]
        if not valid:
            return []

        track_boxes = {
            int(b.id.item()): list(map(int, b.xyxy[0].tolist()))
            for b in valid
        }
        track_confs = {
            int(b.id.item()): float(b.conf.item())
            for b in valid
        }

        # Re-ID setiap N frame
        if self.frame_count % REID_INTERVAL == 0:
            xyxys = np.array(list(track_boxes.values()), dtype=np.float32)
            embs  = self.reid(frame, boxes=xyxys)
            for i, tid in enumerate(track_boxes):
                self.track_to_pid[tid] = self._match_or_create(embs[i])

        tracks = []
        for tid, bbox in track_boxes.items():
            person_id = self.track_to_pid.get(tid, tid)
            tracks.append({
                "person_id": person_id,
                "track_id":  tid,
                "bbox":      bbox,
                "conf":      track_confs[tid],
            })

        return tracks

    def reset(self):
        """Reset gallery dan semua state — pakai saat ganti video/sesi."""
        self.gallery.clear()
        self.track_to_pid.clear()
        self.next_pid    = 1
        self.frame_count = 0
