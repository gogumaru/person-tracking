import cv2
import random
from ultralytics import YOLO

model = YOLO("/Users/benediktaaa/Documents/AML/person-tracking/runs/detect/train-17-v/weights/best.pt")

cap = cv2.VideoCapture("inference/retail_cctv.MP4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

# Set None untuk tampilkan semua ID, atau angka (misal 5) untuk lock sejumlah ID
MAX_IDS = None

# Assign consistent color per track ID
track_colors = {}
hit_count = {}
locked_ids = set()
MIN_HITS = 8

def get_color(track_id):
    if track_id not in track_colors:
        random.seed(track_id)
        track_colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return track_colors[track_id]

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended.")
        break

    results = model.track(
        source=frame,
        conf=0.45,
        tracker="botsort-reid.yaml",
        persist=True,
        verbose=False,
    )

    if results[0].boxes is not None:
        boxes = results[0].boxes
        valid = [box for box in boxes if box.id is not None]

        # update hit count dan kunci ID yang sudah cukup stabil
        if MAX_IDS is not None and len(locked_ids) < MAX_IDS:
            for box in sorted(valid, key=lambda b: float(b.conf.item()), reverse=True):
                track_id = int(box.id.item())
                hit_count[track_id] = hit_count.get(track_id, 0) + 1
                if hit_count[track_id] >= MIN_HITS and track_id not in locked_ids:
                    locked_ids.add(track_id)
                    print(f"Locked ID: {track_id}")
                if len(locked_ids) >= MAX_IDS:
                    break

        # tampilkan semua jika MAX_IDS None, atau hanya locked IDs
        for box in valid:
            track_id = int(box.id.item())
            if MAX_IDS is not None and track_id not in locked_ids:
                continue

            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            color = get_color(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8), font, 0.4, color, 1)

    cv2.imshow("BoT-SORT Tracking", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
