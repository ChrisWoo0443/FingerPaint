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

'''
https://www.youtube.com/watch?v=1H9qUzmSm_M
websocket tutorial
'''

app = FastAPI()
app.mount("/style", StaticFiles(directory="style"), name="style")
templates = Jinja2Templates(directory="template")


'''
change camera_val to your the value used by your webcam
'''
camera_val = 1
camera = cv2.VideoCapture(camera_val)


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

@app.websocket('/canvas')
async def get_canvas(websocket: WebSocket):
    await websocket.accept()
    try:
        draw_mode = False
        touched = False
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                
                frame = cv2.flip(frame, 1)
                canvas = np.full(frame.shape, 255, dtype="uint8")

                Process = hands.process(frame)  # process the image
                landmarkList = []

                # Draw hand landmarks
                if Process.multi_hand_landmarks:
                    for handlm in Process.multi_hand_landmarks:
                        for _id, landmarks in enumerate(handlm.landmark):
                            height, width, color_channels = frame.shape
                            x, y = int(landmarks.x * width), int(landmarks.y * height)
                            landmarkList.append([_id, x, y])
                        Draw.draw_landmarks(canvas, handlm, mpHands.HAND_CONNECTIONS)

                if landmarkList:
                    # Get the coordinates of the landmarks
                    # x_0, y_0 = landmarkList[0][1], landmarkList[0][2]       # palm
                    x_1, y_1 = landmarkList[4][1], landmarkList[4][2]       # tips of thumb
                    x_2, y_2 = landmarkList[8][1], landmarkList[8][2]       # tips of index
                    x_3, y_3 = landmarkList[12][1], landmarkList[12][2]     # tips of middle
                    # x_4, y_4 = landmarkList[16][1], landmarkList[16][2]     # tips of ring
                    # x_5, y_5 = landmarkList[20][1], landmarkList[20][2]     # tips of pinky
                
                     # Draw circles on the tips of each finger
                    cv2.circle(canvas, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)      # thumb
                    cv2.circle(canvas, (x_2, y_2), 7, (0, 0, 0), cv2.FILLED)      # index
                    cv2.circle(canvas, (x_3, y_3), 7, (0, 255, 0), cv2.FILLED)      # middle

                    TI = hypot(x_2-x_1, y_2-y_1)        # thumb and index
                    interTI = np.interp(TI, [15,220], [0,100])
                    TM = hypot(x_3-x_1, y_3-y_1)        # thumb and middle
                    interTM = np.interp(TM, [15, 220], [0, 100])

                    # clear the canvas by touching thumb to pointer
                    if int(interTI) < 20:
                        myPoints.clear()

                    # toggle drawing mode by touching thumb to middle finger
                    if int(interTM) < 20 and not touched:
                        old_draw = draw_mode
                        draw_mode = not draw_mode
                        if old_draw:
                            myPoints.append(None)
                        touched = True
                    elif int(interTM) > 30:
                        touched = False

                    # Draw if in drawing mode
                    if draw_mode:
                        myPoints.append((x_2, y_2, (0, 0, 0), circleRadius))

                # smoother drawing instead of drawing dots
                if len(myPoints) >= 2:
                    for i in range(1, len(myPoints)):
                        if myPoints[i - 1] is None or myPoints[i] is None:
                            continue
                        pt1 = myPoints[i - 1][:2]
                        pt2 = myPoints[i][:2]
                        cv2.line(canvas, pt1, pt2, (0, 0, 0), circleRadius)

                _, buffer = cv2.imencode('.jpg', canvas)

                await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.01)
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")




@app.websocket('/webcam')
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try: 
        draw_mode = False
        touched = False
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                
                frame = cv2.flip(frame, 1)

                Process = hands.process(frame)  # process the image
                landmarkList = []

                # Draw hand landmarks
                if Process.multi_hand_landmarks:
                    for handlm in Process.multi_hand_landmarks:
                        for _id, landmarks in enumerate(handlm.landmark):
                            height, width, color_channels = frame.shape
                            x, y = int(landmarks.x * width), int(landmarks.y * height)
                            landmarkList.append([_id, x, y])
                        Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

                if landmarkList:
                    # Get the coordinates of the landmarks
                    # x_0, y_0 = landmarkList[0][1], landmarkList[0][2]       # palm
                    x_1, y_1 = landmarkList[4][1], landmarkList[4][2]       # tips of thumb
                    x_2, y_2 = landmarkList[8][1], landmarkList[8][2]       # tips of index
                    x_3, y_3 = landmarkList[12][1], landmarkList[12][2]     # tips of middle
                    # x_4, y_4 = landmarkList[16][1], landmarkList[16][2]     # tips of ring
                    # x_5, y_5 = landmarkList[20][1], landmarkList[20][2]     # tips of pinky
                
                     # Draw circles on the tips of each finger
                    # cv2.circle(frame, (x_0, y_0), 7, (0, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)       # thumb
                    cv2.circle(frame, (x_2, y_2), 7, (0, 0, 0), cv2.FILLED)       # index
                    cv2.circle(frame, (x_3, y_3), 7, (0, 255, 0), cv2.FILLED)       # middle
                    # cv2.circle(frame, (x_4, y_4), 7, (0, 255, 0), cv2.FILLED)
                    # cv2.circle(frame, (x_5, y_5), 7, (0, 255, 0), cv2.FILLED)

                    TI = hypot(x_2-x_1, y_2-y_1)        # thumb and index
                    interTI = np.interp(TI, [15,220], [0,100])
                    # MP = hypot(x_3-x_0, y_3-y_0)        # middle and palm
                    # interMP = np.interp(MP, [15,300], [0,100])
                    # RP = hypot(x_4-x_0, y_4-y_0)        # ring and palm
                    # interRP = np.interp(RP, [15,300], [0,100])
                    # PP = hypot(x_5-x_0, y_5-y_0)        # pinky and palm
                    # interPP = np.interp(PP, [15,300], [0,100])
                    TM = hypot(x_3-x_1, y_3-y_1)        # thumb and middle
                    interTM = np.interp(TM, [15, 220], [0, 100])

                    # clear the canvas by touching thumb to pointer
                    if int(interTI) < 20:
                        myPoints.clear()

                    # toggle drawing mode by touching thumb to middle finger
                    if int(interTM) < 20 and not touched:
                        old_draw = draw_mode
                        draw_mode = not draw_mode
                        if old_draw:
                            myPoints.append(None)
                        touched = True
                    elif int(interTM) > 30:
                        touched = False

                    # Draw if in drawing mode
                    if draw_mode:
                        myPoints.append((x_2, y_2, (0, 0, 0), circleRadius))

                # smoother drawing instead of drawing dots
                # if len(myPoints) >= 2:
                #     for i in range(1, len(myPoints)):
                #         if myPoints[i - 1] is None or myPoints[i] is None:
                #             continue
                #         pt1 = myPoints[i - 1][:2]
                #         pt2 = myPoints[i][:2]
                #         cv2.line(frame, pt1, pt2, (0, 0, 0), circleRadius)

                # for points in myPoints:
                #     cv2.circle(canvas, (points[0], points[1]), points[3], points[2], cv2.FILLED)
                #     cv2.circle(frame, (points[0], points[1]), points[3], points[2], cv2.FILLED)

                _, buffer = cv2.imencode('.jpg', frame)

                await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.01)
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")


if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)

