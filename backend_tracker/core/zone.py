import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Zone:
    """
    Satu zone berbentuk polygon.

    points: list of (x, y) normalized 0.0–1.0 terhadap frame size
    contoh: [(0.1, 0.2), (0.4, 0.2), (0.4, 0.6), (0.1, 0.6)]
    """
    zone_id:   int
    name:      str
    points:    list[tuple[float, float]]   # normalized coordinates
    direction: Literal["in", "out", "both"] = "both"

    # state internal — track siapa yang sedang di dalam zone
    _inside: set[int] = field(default_factory=set, repr=False)

    # counter
    count_in:  int = 0
    count_out: int = 0

    def to_pixel(self, w: int, h: int) -> np.ndarray:
        """Convert normalized points ke pixel coordinates."""
        return np.array(
            [(int(x * w), int(y * h)) for x, y in self.points],
            dtype=np.int32,
        )


class ZoneManager:
    """
    Kelola semua zone dan deteksi crossing per frame.
    """

    def __init__(self):
        self.zones: dict[int, Zone] = {}

    # ── Zone CRUD ─────────────────────────────────────────────────────────────

    def add_zone(self, zone_id: int, name: str,
                 points: list[tuple[float, float]],
                 direction: str = "both") -> Zone:
        zone = Zone(zone_id=zone_id, name=name,
                    points=points, direction=direction)
        self.zones[zone_id] = zone
        return zone

    def remove_zone(self, zone_id: int):
        self.zones.pop(zone_id, None)

    def update_zone_points(self, zone_id: int,
                           points: list[tuple[float, float]]):
        if zone_id in self.zones:
            self.zones[zone_id].points = points
            self.zones[zone_id]._inside.clear()

    def get_all(self) -> list[Zone]:
        return list(self.zones.values())

    # ── Geometry ──────────────────────────────────────────────────────────────

    @staticmethod
    def _point_in_polygon(px: float, py: float,
                          poly: list[tuple[float, float]]) -> bool:
        """Ray casting algorithm — cek apakah titik (px,py) di dalam polygon."""
        n = len(poly)
        inside = False
        x, y = px, py
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _bbox_center(bbox: list[int]) -> tuple[float, float]:
        """Ambil titik bawah-tengah bbox (lebih akurat untuk posisi kaki orang)."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)  # center-bottom

    # ── Update per frame ──────────────────────────────────────────────────────

    def update(self, tracks: list[dict],
               frame_w: int, frame_h: int) -> list[dict]:
        """
        Cek setiap track terhadap semua zone.

        Args:
            tracks: output dari PersonTracker.update()
            frame_w, frame_h: dimensi frame (untuk denormalize zone points)

        Returns:
            list of crossing events:
            {
                "zone_id":   int,
                "zone_name": str,
                "person_id": int,
                "event":     "enter" | "exit",
            }
        """
        events = []

        for zone in self.zones.values():
            # normalisasi titik zone sudah di 0-1, cocokkan dengan center bbox
            # yang juga kita normalisasi
            for track in tracks:
                pid  = track["person_id"]
                bbox = track["bbox"]

                # center dalam koordinat normalized
                cx_px, cy_px = self._bbox_center(bbox)
                cx = cx_px / frame_w
                cy = cy_px / frame_h

                is_inside = self._point_in_polygon(cx, cy, zone.points)
                was_inside = pid in zone._inside

                if is_inside and not was_inside:
                    # orang baru masuk zone
                    zone._inside.add(pid)
                    zone.count_in += 1
                    if zone.direction in ("in", "both"):
                        events.append({
                            "zone_id":   zone.zone_id,
                            "zone_name": zone.name,
                            "person_id": pid,
                            "event":     "enter",
                        })

                elif not is_inside and was_inside:
                    # orang keluar zone
                    zone._inside.discard(pid)
                    zone.count_out += 1
                    if zone.direction in ("out", "both"):
                        events.append({
                            "zone_id":   zone.zone_id,
                            "zone_name": zone.name,
                            "person_id": pid,
                            "event":     "exit",
                        })

        return events

    def get_stats(self) -> list[dict]:
        """Snapshot count semua zone untuk dikirim ke macOS app."""
        return [
            {
                "zone_id":    z.zone_id,
                "name":       z.name,
                "count_in":   z.count_in,
                "count_out":  z.count_out,
                "current":    len(z._inside),  # orang yang sedang di dalam
            }
            for z in self.zones.values()
        ]

    def reset_counts(self, zone_id: int | None = None):
        """Reset counter. zone_id=None → reset semua zone."""
        targets = [self.zones[zone_id]] if zone_id else self.zones.values()
        for z in targets:
            z.count_in  = 0
            z.count_out = 0
            z._inside.clear()
