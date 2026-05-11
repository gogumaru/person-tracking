import asyncio
import websockets
import cv2
import numpy as np

async def test():
    async with websockets.connect("ws://localhost:8000/stream/ws") as ws:
        while True:
            data = await ws.recv()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("Stream Test", frame)
            if cv2.waitKey(1) == ord("q"):
                break

asyncio.run(test())
