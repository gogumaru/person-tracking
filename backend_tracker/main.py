import asyncio
import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT
from core.tracker import PersonTracker
from core.zone import ZoneManager
from core.trail import TrailManager
from storage.database import init_db, start_session, end_session, load_all_zones, log_crossing
from api import stream, zones, stats


# ── State global (shared antar routes) ───────────────────────────────────────
tracker    = PersonTracker()
zone_mgr   = ZoneManager()
trail_mgr  = TrailManager()
session_id: int | None = None
cap:        cv2.VideoCapture | None = None


# ── Lifespan (startup & shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_id, cap

    # Startup
    init_db()

    # Restore zone dari database
    for z in load_all_zones():
        zone_mgr.add_zone(**z)

    # Buka kamera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    session_id = start_session()
    print(f"Session started: {session_id}")

    # Mulai background loop tracking
    asyncio.create_task(tracking_loop())

    yield

    # Shutdown
    if session_id:
        end_session(session_id)
    if cap:
        cap.release()
    print("Session ended.")


# ── Background tracking loop ──────────────────────────────────────────────────
latest_frame: np.ndarray | None = None
latest_tracks: list[dict]       = []
latest_events: list[dict]       = []


async def tracking_loop():
    global latest_frame, latest_tracks, latest_events

    while True:
        if cap is None or not cap.isOpened():
            await asyncio.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.05)
            continue

        # Jalankan tracking di thread pool supaya tidak block event loop
        loop = asyncio.get_event_loop()
        tracks = await loop.run_in_executor(None, tracker.update, frame)

        h, w = frame.shape[:2]
        events = zone_mgr.update(tracks, frame_w=w, frame_h=h)

        # Simpan crossing events ke DB
        for ev in events:
            log_crossing(
                session_id=session_id,
                zone_id=ev["zone_id"],
                person_id=ev["person_id"],
                event=ev["event"],
            )

        trail_mgr.record(tracks)

        latest_frame  = frame
        latest_tracks = tracks
        latest_events = events

        await asyncio.sleep(0)   # yield ke event loop


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Person Tracker API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ganti ke origin macOS app saat production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stream.router, prefix="/stream", tags=["stream"])
app.include_router(zones.router,  prefix="/zones",  tags=["zones"])
app.include_router(stats.router,  prefix="/stats",  tags=["stats"])


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "session_id": session_id,
        "zones":      len(zone_mgr.zones),
    }
