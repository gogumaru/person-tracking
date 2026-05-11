import cv2
import random
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from boxmot.trackers import StrongSort

# Detector — pretrained YOLO, class 0 = person
detector = YOLO("yolo26n.pt")
# detector = YOLO("/Users/benediktaaa/Documents/AML/person-tracking/runs/detect/train-17-v/weights/best.pt")  # model custom hasil training

# StrongSORT + OSNet Re-ID
# osnet_x0_25_msmt17.pt akan auto-download pertama kali
tracker = StrongSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device=torch.device("mps"),
    half=False,
    max_cos_dist=0.3,
    max_iou_dist=0.7,
    n_init=3,
    nn_budget=200,
    max_age=180,   # frame sebelum track dihapus saat orang hilang (default 30 ~1dtk, 90 ~3dtk)
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

track_colors = {}

def get_color(track_id):
    if track_id not in track_colors:
        random.seed(int(track_id))
        track_colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return track_colors[track_id]

font = cv2.FONT_HERSHEY_COMPLEX
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame.")
        break

    # Deteksi hanya class person (0)
    results = detector.predict(
        source=frame,
        conf=0.45,
        classes=0,
        verbose=False,
    )

    # Konversi deteksi ke format boxmot: [x1, y1, x2, y2, conf, cls]
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        dets = np.column_stack([
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        ])
    else:
        dets = np.empty((0, 6))

    # Update tracker — returns [x1, y1, x2, y2, track_id, conf, cls, idx]
    tracks = tracker.update(dets, frame)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t[:4])
        track_id = int(t[4])
        conf = float(t[5])

        color = get_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {conf:.2f}", (x1, y1 - 8), font, 0.4, color, 1)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), font, 0.6, (0, 255, 0), 1)

    cv2.imshow("StrongSORT + OSNet Re-ID", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
