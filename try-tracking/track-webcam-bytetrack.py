import cv2
import random
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from boxmot.reid.core.reid import ReID

# YOLO + ByteTrack untuk tracking frame-to-frame
# model = Y
# OLO("yolo26n.pt")
model = YOLO("runs/detect/train-17-v/weights/best.pt")

# CLIP sebagai Re-ID model untuk recovery setelah occlusion
reid_model = ReID(
    path=Path("clip_market1501.pt"),
    device=torch.device("mps"),
    half=False,
)

SIMILARITY_THRESH = 0.6   # cosine similarity minimum untuk dianggap orang sama
MAX_GALLERY_PER_ID = 10
REID_INTERVAL = 5         # ekstrak embedding setiap N frame

# Gallery: person_id → list of CLIP embeddings
gallery = {}
next_person_id = 1

# Map ByteTrack track_id → person_id (persistent)
track_to_person = {}

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def match_or_create(embedding):
    global next_person_id
    best_id, best_sim = None, -1.0

    for pid, embeddings in gallery.items():
        sim = cosine_similarity(embedding, np.mean(embeddings, axis=0))
        if sim > best_sim:
            best_sim = sim
            best_id = pid

    if best_id is not None and best_sim >= SIMILARITY_THRESH:
        gallery[best_id].append(embedding)
        if len(gallery[best_id]) > MAX_GALLERY_PER_ID:
            gallery[best_id].pop(0)
        return best_id, best_sim
    else:
        pid = next_person_id
        next_person_id += 1
        gallery[pid] = [embedding]
        return pid, best_sim

track_colors = {}

def get_color(pid):
    if pid not in track_colors:
        random.seed(int(pid))
        track_colors[pid] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return track_colors[pid]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

font = cv2.FONT_HERSHEY_COMPLEX
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame.")
        break

    frame_count += 1
    do_reid = (frame_count % REID_INTERVAL == 0)

    results = model.track(
        source=frame,
        conf=0.45,
        classes=0,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )

    if results[0].boxes is not None:
        boxes = results[0].boxes
        valid = [b for b in boxes if b.id is not None]

        if do_reid and len(valid) > 0:
            # Kumpulkan semua bbox sekaligus untuk batch inference
            xyxys = np.array([b.xyxy[0].tolist() for b in valid])
            embeddings = reid_model(frame, boxes=xyxys)

            for i, box in enumerate(valid):
                track_id = int(box.id.item())
                emb = embeddings[i]
                pid, _ = match_or_create(emb)
                track_to_person[track_id] = pid

        for box in valid:
            track_id = int(box.id.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            pid = track_to_person.get(track_id)
            if pid is not None:
                color = get_color(pid)
                label = f"ID:{pid} {conf:.2f}"
            else:
                color = (128, 128, 128)
                label = f"TID:{track_id} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), font, 0.4, color, 1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), font, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"Identities: {len(gallery)}", (10, 55), font, 0.6, (0, 255, 0), 1)

    cv2.imshow("ByteTrack + CLIP Re-ID", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
