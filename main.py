import cv2
import mediapipe as mp
import numpy as np
from math import hypot

from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from websockets.exceptions import ConnectionClosed

import asyncio
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="template")

camera = cv2.VideoCapture(1)
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



@app.websocket('/websocket')
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try: 
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                
                #







                _, buffer = cv2.imencode('.jpg', frame)
                await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.01)
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")




if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)