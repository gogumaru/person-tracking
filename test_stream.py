import asyncio
import threading
import websockets
import cv2
import numpy as np
import requests

BACKEND = "http://localhost:8000"
_click_queue: list[tuple[int, int]] = []
_lock = threading.Lock()


def _on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        with _lock:
            _click_queue.append((x, y))


def _send_click(x: int, y: int):
    try:
        requests.post(f"{BACKEND}/stream/click", json={"x": x, "y": y}, timeout=1)
    except Exception:
        pass


async def test():
    cv2.namedWindow("Stream Test")
    cv2.setMouseCallback("Stream Test", _on_mouse)

    async with websockets.connect("ws://localhost:8000/stream/ws") as ws:
        while True:
            data = await ws.recv()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("Stream Test", frame)

            # Drain pending clicks (non-blocking, fire-and-forget)
            with _lock:
                pending = list(_click_queue)
                _click_queue.clear()
            for x, y in pending:
                threading.Thread(target=_send_click, args=(x, y), daemon=True).start()

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("c"):   # 'c' to clear trail manually
                try:
                    requests.delete(f"{BACKEND}/stream/trail", timeout=1)
                except Exception:
                    pass

    cv2.destroyAllWindows()


asyncio.run(test())
