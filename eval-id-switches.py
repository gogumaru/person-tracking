"""
ID Switch Evaluator — compare tracker dengan/tanpa CLIP Re-ID

MODE:
  "bytetrack"       → ByteTrack saja (baseline)
  "bytetrack_clip"  → ByteTrack + CLIP Re-ID
  "botsort"         → BoT-SORT + Re-ID bawaan
"""

import cv2
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH   = 0 #"inference/mall_cctv.MP4"   # ganti ke 0 untuk webcam
MODEL_PATH   = "yolo26n.pt"
CONF         = 0.45
IOU_THRESH   = 0.3    # IoU minimum untuk dianggap orang yang sama
WINDOW_FRAME = 90     # frame window re-entry (~1 detik di 30fps)

MODE = "bytetrack_clip"    # pilih: "bytetrack" | "bytetrack_clip" | "botsort" | "botsort_osnet"

# Re-ID config (aktif kalau MODE == "bytetrack_clip" atau "botsort_osnet")
CLIP_SIMILARITY_THRESH = 0.6
CLIP_REID_INTERVAL     = 5
CLIP_MAX_GALLERY       = 10
# ──────────────────────────────────────────────────────────────────────────────

# Setup tracker config berdasarkan mode
if MODE == "bytetrack":
    TRACKER   = "bytetrack.yaml"
    USE_REID  = False
    REID_WEIGHTS = None
elif MODE == "bytetrack_clip":
    TRACKER   = "bytetrack.yaml"
    USE_REID  = True
    REID_WEIGHTS = "clip_market1501.pt"
elif MODE == "botsort":
    TRACKER   = "botsort.yaml"
    USE_REID  = False
    REID_WEIGHTS = None
elif MODE == "botsort_osnet":
    TRACKER   = "botsort.yaml"
    USE_REID  = True
    REID_WEIGHTS = "osnet_x0_25_msmt17.pt"
else:
    raise ValueError(f"MODE tidak dikenal: {MODE}")

print(f"Mode: {MODE.upper()}")
print(f"Tracker: {TRACKER} | Re-ID: {REID_WEIGHTS or 'None'}")
print("-" * 50)

# Load model
model = YOLO(MODEL_PATH)

# Load Re-ID model kalau dipakai
reid_model = None
if USE_REID:
    import torch
    from boxmot.reid.core.reid import ReID
    reid_model = ReID(
        path=Path(REID_WEIGHTS),
        device=torch.device("mps"),
        half=False,
    )
    gallery       = {}
    next_pid      = 1
    track_to_pid  = {}

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def match_or_create(emb):
        global next_pid
        best_id, best_sim = None, -1.0
        for pid, embs in gallery.items():
            sim = cosine_sim(emb, np.mean(embs, axis=0))
            if sim > best_sim:
                best_sim, best_id = sim, pid
        if best_id is not None and best_sim >= CLIP_SIMILARITY_THRESH:
            gallery[best_id].append(emb)
            if len(gallery[best_id]) > CLIP_MAX_GALLERY:
                gallery[best_id].pop(0)
            return best_id
        pid = next_pid; next_pid += 1
        gallery[pid] = [emb]
        return pid

# Buka video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Tidak bisa buka: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30

# ── State ─────────────────────────────────────────────────────────────────────
frame_count    = 0
active_tracks  = {}          # {display_id: bbox}
dead_tracks    = {}          # {display_id: (bbox, frame_died)}
track_lifetime = defaultdict(int)
all_ids_seen   = set()
id_switches    = []
track_colors   = {}

def get_color(tid):
    if tid not in track_colors:
        random.seed(int(tid))
        track_colors[tid] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return track_colors[tid]

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

font = cv2.FONT_HERSHEY_COMPLEX

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        results = model.track(
            source=frame, conf=CONF, classes=0,
            tracker=TRACKER, persist=True, verbose=False,
        )

        current_boxes = {}
        if results[0].boxes is not None:
            valid = [b for b in results[0].boxes if b.id is not None]
            for box in valid:
                tid  = int(box.id.item())
                bbox = list(map(int, box.xyxy[0].tolist()))
                current_boxes[tid] = bbox

        # Re-ID: map track_id → person_id
        if USE_REID and frame_count % CLIP_REID_INTERVAL == 0 and current_boxes:
            xyxys = np.array([current_boxes[t] for t in current_boxes], dtype=np.float32)
            embs  = reid_model(frame, boxes=xyxys)
            for i, tid in enumerate(current_boxes):
                pid = match_or_create(embs[i])
                track_to_pid[tid] = pid

        display_boxes = {}
        for tid, bbox in current_boxes.items():
            did = track_to_pid.get(tid, tid) if USE_REID else tid
            display_boxes[did] = bbox
            all_ids_seen.add(did)
            track_lifetime[did] += 1

        # Deteksi ID switch
        new_ids  = set(display_boxes) - set(active_tracks)
        lost_ids = set(active_tracks) - set(display_boxes)

        for new_did in new_ids:
            new_bbox = display_boxes[new_did]
            for dead_did, (dead_bbox, frame_died) in list(dead_tracks.items()):
                if frame_count - frame_died > WINDOW_FRAME:
                    continue
                if iou(new_bbox, dead_bbox) >= IOU_THRESH:
                    id_switches.append((frame_count, dead_did, new_did))
                    print(f"  [Frame {frame_count:4d}] ID SWITCH: {dead_did} → {new_did}  "
                          f"(IoU={iou(new_bbox, dead_bbox):.2f})")

        for lost_did in lost_ids:
            dead_tracks[lost_did] = (active_tracks[lost_did], frame_count)

        dead_tracks = {d: v for d, v in dead_tracks.items()
                       if frame_count - v[1] <= WINDOW_FRAME}

        active_tracks = display_boxes

        # Visualisasi
        for did, bbox in display_boxes.items():
            x1, y1, x2, y2 = bbox
            color = get_color(did)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{did}", (x1, y1 - 6), font, 0.4, color, 1)

        cv2.putText(frame, f"[{MODE}]", (10, 25), font, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"ID Switches: {len(id_switches)}", (10, 50), font, 0.55, (0, 100, 255), 1)
        cv2.putText(frame, f"Total IDs: {len(all_ids_seen)}", (10, 75), font, 0.55, (255, 200, 0), 1)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 100), font, 0.55, (0, 255, 0), 1)

        cv2.imshow(f"ID Switch Eval — {MODE}", frame)
        if cv2.waitKey(1) == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[Dihentikan manual]")
finally:
    cap.release()
    cv2.destroyAllWindows()

# ── Laporan ───────────────────────────────────────────────────────────────────
avg_life = sum(track_lifetime.values()) / len(track_lifetime) if track_lifetime else 0

print("\n" + "=" * 50)
print(f"HASIL  [{MODE.upper()}]")
print("=" * 50)
print(f"Total frame        : {frame_count}")
print(f"Total unique ID    : {len(all_ids_seen)}")
print(f"ID switches        : {len(id_switches)}")
print(f"Avg track lifetime : {avg_life:.1f} frame ({avg_life/fps:.1f} detik)")
print("=" * 50)
