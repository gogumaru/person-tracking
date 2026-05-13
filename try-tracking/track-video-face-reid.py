import cv2
import time
import random

import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

VIDEO_PATH = "inference/mall_cctv.MP4"

detector = YOLO("yolo26n.pt")

app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))

gallery = {}
next_face_id = 1
SIMILARITY_THRESH = 0.4
MAX_GALLERY_PER_ID = 10
track_to_face = {}
REID_INTERVAL = 5

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def match_or_create(embedding):
    global next_face_id
    best_id, best_sim = None, -1.0

    for fid, embeddings in gallery.items():
        sim = cosine_similarity(embedding, np.mean(embeddings, axis=0))
        if sim > best_sim:
            best_sim = sim
            best_id = fid

    if best_id is not None and best_sim >= SIMILARITY_THRESH:
        gallery[best_id].append(embedding)
        if len(gallery[best_id]) > MAX_GALLERY_PER_ID:
            gallery[best_id].pop(0)
        return best_id, best_sim
    else:
        fid = next_face_id
        next_face_id += 1
        gallery[fid] = [embedding]
        return fid, best_sim

track_colors = {}

def get_color(fid):
    if fid not in track_colors:
        random.seed(int(fid))
        track_colors[fid] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return track_colors[fid]

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Cannot open video: {VIDEO_PATH}")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
frame_delay = max(1, int(1000 / fps_video))

font = cv2.FONT_HERSHEY_COMPLEX
frame_count = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_count += 1
    do_reid = (frame_count % REID_INTERVAL == 0)
    h, w = frame.shape[:2]

    results = detector.track(
        source=frame,
        conf=0.45,
        classes=0,
        tracker="botsort.yaml",
        persist=True,
        verbose=False,
    )

    if results[0].boxes is not None:
        boxes = results[0].boxes
        valid = [b for b in boxes if b.id is not None]

        for box in valid:
            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if do_reid:
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    faces = app.get(crop)
                    if faces:
                        best_face = max(faces, key=lambda f: f.det_score)
                        if best_face.det_score >= 0.5:
                            face_id, _ = match_or_create(best_face.embedding)
                            track_to_face[track_id] = face_id

            face_id = track_to_face.get(track_id)
            if face_id is not None:
                label = f"ID:{face_id}"
                color = get_color(face_id)
            else:
                label = f"TID:{track_id}"
                color = (128, 128, 128)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), font, 0.4, color, 1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), font, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"Identities: {len(gallery)}", (10, 55), font, 0.6, (0, 255, 0), 1)

    cv2.imshow("Video + Face Re-ID (ArcFace)", frame)

    if cv2.waitKey(frame_delay) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
