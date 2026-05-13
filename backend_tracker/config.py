from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"

YOLO_MODEL  = MODELS_DIR / "best-4.pt"
REID_WEIGHTS = MODELS_DIR / "osnet_x0_25_msmt17.pt"

DB_PATH     = BASE_DIR / "storage" / "tracker.db"

# ── Camera / Video ────────────────────────────────────────────────────────────
CAMERA_SOURCE = 0 #"rtsp://192.168.92.180:5543/a66abb5684c45962d887564f08346e8d/live/channel0"          # 0 = webcam, atau path string ke file video
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

# ── Detection ─────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25      # confidence minimum deteksi person
TRACKER        = str(BASE_DIR / "botsort.yaml")

# ── Re-ID ─────────────────────────────────────────────────────────────────────
REID_SIMILARITY_THRESH = 0.6   # cosine similarity minimum untuk match orang
REID_INTERVAL          = 5     # ekstrak embedding setiap N frame
REID_MAX_GALLERY       = 10    # max embedding tersimpan per orang

# ── Zone ──────────────────────────────────────────────────────────────────────
# Koordinat zone disimpan sebagai normalized (0.0–1.0) terhadap frame size
# supaya tidak pecah kalau resolusi kamera berubah
ZONE_CROSS_DIRECTION = "both"  # "in", "out", atau "both"
ZONE_EXIT_THRESHOLD = 25

# ── FastAPI / Streaming ───────────────────────────────────────────────────────
HOST        = "0.0.0.0"
PORT        = 8000
STREAM_FPS  = 25           # target FPS untuk WebSocket stream ke macOS app
JPEG_QUALITY = 80          # kualitas kompresi frame yang dikirim (1-100)
