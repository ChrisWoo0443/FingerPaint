import cv2
import mediapipe as mp
import numpy as np
from math import hypot

from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from websockets.exceptions import ConnectionClosed

import asyncio
import uvicorn

import base64
from io import BytesIO
from PIL import Image

'''
https://www.youtube.com/watch?v=1H9qUzmSm_M
websocket tutorial
'''

app = FastAPI()
app.mount("/style", StaticFiles(directory="style"), name="style")
templates = Jinja2Templates(directory="template")


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,        # specify whether an image is static or it is a video stream
    model_complexity=1,             # complexity of the hand landmark model: 0 or 1
    min_detection_confidence=0.75,  # minimum confidence, 0-1 default is 0.5
    min_tracking_confidence=0.75,   # minimum confidence, 0-1 default is 0.5
    max_num_hands=1)                # maximum number of hands default is 2

Draw = mp.solutions.drawing_utils

# drawing points and circle size
myPoints = []

circleRadius = 10

# switches
draw_mode = False
touched = False


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.websocket("/process")
async def process_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            header, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)

            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)

            # Process with Mediapipe
            Process = hands.process(frame)
            if Process.multi_hand_landmarks:
                for handlm in Process.multi_hand_landmarks:
                    Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

            _, buffer = cv2.imencode('.jpg', frame)
            base64_img = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(f"data:image/jpeg;base64,{base64_img}")
            await asyncio.sleep(0.01)
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")


@app.websocket('/canvas')
async def canvas_from_client(websocket: WebSocket):
    await websocket.accept()
    draw_mode = False
    touched = False
    try:
        while True:
            data = await websocket.receive_text()
            _, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)

            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)

            canvas = np.full(frame.shape, 255, dtype="uint8")
            Process = hands.process(frame)
            landmarkList = []

            if Process.multi_hand_landmarks:
                for handlm in Process.multi_hand_landmarks:
                    for _id, landmarks in enumerate(handlm.landmark):
                        h, w, _ = frame.shape
                        x, y = int(landmarks.x * w), int(landmarks.y * h)
                        landmarkList.append([_id, x, y])
                    Draw.draw_landmarks(canvas, handlm, mpHands.HAND_CONNECTIONS)

            if landmarkList:
                # Get the coordinates of the landmarks
                # x0, y0 = landmarkList[0][1], landmarkList[0][2]       # palm
                x1, y1 = landmarkList[4][1], landmarkList[4][2]       # tips of thumb
                x2, y2 = landmarkList[8][1], landmarkList[8][2]       # tips of index
                x3, y3 = landmarkList[12][1], landmarkList[12][2]     # tips of middle
                # x4, y4 = landmarkList[16][1], landmarkList[16][2]     # tips of ring
                # x5, y5 = landmarkList[20][1], landmarkList[20][2]     # tips of pinky

                # Draw circles on the tips of each finger
                cv2.circle(canvas, (x1, y1), 7, (0, 255, 0), cv2.FILLED)      # thumb
                cv2.circle(canvas, (x2, y2), 7, (0, 0, 0), cv2.FILLED)      # index
                cv2.circle(canvas, (x3, y3), 7, (0, 255, 0), cv2.FILLED)      # middle

                TI = hypot(x2 - x1, y2 - y1)        # thumb and index
                TM = hypot(x3 - x1, y3 - y1)        # thumb and middle
                interTI = np.interp(TI, [15, 220], [0, 100])
                interTM = np.interp(TM, [15, 220], [0, 100])

                # clear the canvas by touching thumb to pointer
                if int(interTI) < 20:
                    myPoints.clear()

                # toggle drawing mode by touching thumb to middle finger
                if int(interTM) < 20 and not touched:
                    draw_mode = not draw_mode
                    if not draw_mode:
                        myPoints.append(None)
                    touched = True
                elif int(interTM) > 30:
                    touched = False

                # Draw if in drawing mode
                if draw_mode:
                    myPoints.append((x2, y2, (0, 0, 0), circleRadius))


            # smoother drawing instead of drawing dots
            if len(myPoints) >= 2:
                for i in range(1, len(myPoints)):
                    if myPoints[i - 1] is None or myPoints[i] is None:
                        continue
                    pt1 = myPoints[i - 1][:2]
                    pt2 = myPoints[i][:2]
                    cv2.line(canvas, pt1, pt2, (0, 0, 0), circleRadius)

            _, buffer = cv2.imencode('.jpg', canvas)
            base64_img = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(f"data:image/jpeg;base64,{base64_img}")
            await asyncio.sleep(0.01)

    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")



if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)

