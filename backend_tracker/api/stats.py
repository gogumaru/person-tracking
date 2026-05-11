from fastapi import APIRouter, Query
from storage.database import get_zone_stats, get_crossings

router = APIRouter()


@router.get("/zones")
def zone_stats(session_id: int | None = Query(None)):
    """
    Jumlah masuk/keluar per zone.
    Kalau session_id tidak diisi, ambil semua data historis.
    """
    return get_zone_stats(session_id=session_id)


@router.get("/zones/live")
def zone_stats_live():
    """
    Count live dari ZoneManager (in-memory) —
    lebih cepat dari DB karena tidak query disk.
    Gunakan ini untuk update real-time di macOS app.
    """
    import main as app_state
    return app_state.zone_mgr.get_stats()


@router.get("/crossings")
def crossing_history(
    zone_id:    int | None = Query(None),
    session_id: int | None = Query(None),
    event:      str | None = Query(None, description="'enter' atau 'exit'"),
):
    """
    Log historis crossing events.
    Bisa difilter by zone, session, atau tipe event.

    Contoh:
      GET /stats/crossings?zone_id=1&event=enter
    """
    return get_crossings(
        zone_id=zone_id,
        session_id=session_id,
        event=event,
    )


@router.get("/summary")
def summary():
    """
    Ringkasan sesi sekarang — untuk dashboard macOS app.
    """
    import main as app_state

    total_people = len({
        t["person_id"] for t in app_state.latest_tracks
    })

    return {
        "session_id":    app_state.session_id,
        "people_now":    total_people,
        "zone_stats":    app_state.zone_mgr.get_stats(),
        "active_tracks": app_state.latest_tracks,
    }
