class TrailManager:
    """
    Simpan history posisi center-bottom bbox per person_id.
    Hanya rekam trail untuk active_id yang sedang dipilih user.
    """

    def __init__(self):
        self._trails: dict[int, list] = {}   # person_id → list of (x, y) pixel
        self.active_id: int | None = None    # person_id yang sedang ditampilkan trailnya

    def set_active(self, person_id: int):
        """Aktifkan trail untuk person_id ini. Reset trail lama kalau beda orang."""
        if self.active_id != person_id:
            self._trails[person_id] = []
        self.active_id = person_id

    def clear(self):
        """Hapus trail aktif."""
        self.active_id = None
        self._trails.clear()

    def record(self, tracks: list[dict]):
        """
        Dipanggil setiap frame — rekam posisi person yang sedang aktif.
        tracks: output dari PersonTracker.update()
        """
        if self.active_id is None:
            return

        for t in tracks:
            if t["person_id"] == self.active_id:
                x1, y1, x2, y2 = t["bbox"]
                cx = (x1 + x2) // 2
                cy = y2   # titik bawah-tengah (posisi kaki)
                if self.active_id not in self._trails:
                    self._trails[self.active_id] = []
                self._trails[self.active_id].append((cx, cy))
                break

    def get_trail(self) -> list[tuple[int, int]]:
        """Return list titik trail untuk active_id."""
        if self.active_id is None:
            return []
        return list(self._trails.get(self.active_id, []))

    def hit_test(self, click_x: int, click_y: int,
                 tracks: list[dict]) -> int | None:
        """
        Cek apakah koordinat klik mengenai bbox seseorang.
        Return person_id kalau kena, None kalau tidak.
        """
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                return t["person_id"]
        return None
