from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from storage.database import save_zone, update_zone, delete_zone, load_all_zones

router = APIRouter()


class ZoneIn(BaseModel):
    name:      str
    points:    list[list[float]]   # [[x, y], ...] normalized 0.0–1.0
    direction: str = "both"        # "in" | "out" | "both"


@router.get("/")
def get_zones():
    """Ambil semua zone yang tersimpan."""
    return load_all_zones()


@router.post("/")
def create_zone(body: ZoneIn):
    """
    Buat zone baru dari macOS app.

    macOS app kirim koordinat polygon yang sudah digambar user,
    dalam format normalized (0.0–1.0) terhadap ukuran frame.
    """
    import main as app_state

    if len(body.points) < 3:
        raise HTTPException(400, "Zone butuh minimal 3 titik.")

    zone_id = save_zone(body.name, body.points, body.direction)

    # Langsung aktifkan di ZoneManager tanpa restart
    app_state.zone_mgr.add_zone(
        zone_id=zone_id,
        name=body.name,
        points=[tuple(p) for p in body.points],
        direction=body.direction,
    )

    return {"zone_id": zone_id, "message": "Zone created."}


@router.put("/{zone_id}")
def edit_zone(zone_id: int, body: ZoneIn):
    """Update nama, titik, atau direction sebuah zone."""
    import main as app_state

    if zone_id not in app_state.zone_mgr.zones:
        raise HTTPException(404, f"Zone {zone_id} tidak ditemukan.")

    if len(body.points) < 3:
        raise HTTPException(400, "Zone butuh minimal 3 titik.")

    update_zone(zone_id, body.name, body.points, body.direction)

    zone = app_state.zone_mgr.zones[zone_id]
    zone.name      = body.name
    zone.direction = body.direction
    app_state.zone_mgr.update_zone_points(
        zone_id, [tuple(p) for p in body.points]
    )

    return {"zone_id": zone_id, "message": "Zone updated."}


@router.delete("/{zone_id}")
def remove_zone(zone_id: int):
    """Hapus zone dari DB dan dari ZoneManager."""
    import main as app_state

    if zone_id not in app_state.zone_mgr.zones:
        raise HTTPException(404, f"Zone {zone_id} tidak ditemukan.")

    delete_zone(zone_id)
    app_state.zone_mgr.remove_zone(zone_id)

    return {"zone_id": zone_id, "message": "Zone deleted."}


@router.post("/{zone_id}/reset")
def reset_zone_count(zone_id: int):
    """Reset counter masuk/keluar sebuah zone."""
    import main as app_state

    if zone_id not in app_state.zone_mgr.zones:
        raise HTTPException(404, f"Zone {zone_id} tidak ditemukan.")

    app_state.zone_mgr.reset_counts(zone_id)
    return {"zone_id": zone_id, "message": "Count reset."}
