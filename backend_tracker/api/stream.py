import cv2
import numpy as np
import asyncio
import random
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import STREAM_FPS, JPEG_QUALITY

router = APIRouter()

# Warna per person_id
_colors: dict[int, tuple] = {}

def _get_color(pid: int) -> tuple:
    if pid not in _colors:
        random.seed(pid)
        _colors[pid] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return _colors[pid]


def _draw(frame: np.ndarray, tracks: list[dict],
          zone_mgr) -> np.ndarray:
    """Gambar bbox tracks + overlay zone ke frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_COMPLEX

    # Gambar zone polygons
    for zone in zone_mgr.get_all():
        pts = zone.to_pixel(w, h)
        cv2.polylines(out, [pts], isClosed=True,
                      color=(0, 255, 255), thickness=2)
        # Label zone
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(out, zone.name, (cx - 30, cy),
                    font, 0.45, (0, 255, 255), 1)

    # Gambar bbox per orang
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        pid   = t["person_id"]
        conf  = t["conf"]
        color = _get_color(pid)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"ID:{pid} {conf:.2f}",
                    (x1, y1 - 6), font, 0.4, color, 1)

    return out


@router.websocket("/ws")
async def video_stream(websocket: WebSocket):
    """
    WebSocket endpoint untuk streaming video ke macOS app.

    macOS app connect ke: ws://localhost:8000/stream/ws

    Server kirim frame sebagai JPEG bytes setiap 1/STREAM_FPS detik.
    macOS app decode bytes → tampilkan sebagai gambar.
    """
    # Import state dari main (late import untuk hindari circular)
    import main as app_state

    await websocket.accept()
    interval = 1.0 / STREAM_FPS

    try:
        while True:
            frame  = app_state.latest_frame
            tracks = app_state.latest_tracks

            if frame is not None:
                annotated = _draw(frame, tracks, app_state.zone_mgr)

                # Encode ke JPEG
                _, buf = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                await websocket.send_bytes(buf.tobytes())

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass


@router.get("/stats-overlay")
def current_overlay():
    """
    Snapshot tracks + zone stats saat ini — untuk polling ringan
    tanpa WebSocket (opsional).
    """
    import main as app_state

    return {
        "tracks":     app_state.latest_tracks,
        "zone_stats": app_state.zone_mgr.get_stats(),
    }
