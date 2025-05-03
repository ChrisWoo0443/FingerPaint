import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import osascript

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,        # specify whether an image is static or it is a video stream
    model_complexity=1,             # complexity of the hand landmark model: 0 or 1
    min_detection_confidence=0.75,  # minimum confidence, 0-1 default is 0.5
    min_tracking_confidence=0.75,   # minimum confidence, 0-1 default is 0.5
    max_num_hands=1)                # maximum number of hands default is 2

Draw = mp.solutions.drawing_utils

st.title("Webcam Live Feed")

# Create a placeholder
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(2)

# Set a stop button
run = st.checkbox('Run Webcam')

while run:
    # Read frame from camera
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to capture image from camera.")
        break

    # Convert BGR (OpenCV) to RGB (Streamlit expects RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    Process = hands.process(frame)  # process the image
    landmarkList = []

    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, color_channels = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    if landmarkList:
        x_0, y_0 = landmarkList[0][1], landmarkList[0][2]       # palm
        x_1, y_1 = landmarkList[4][1], landmarkList[4][2]       # tips of thumb
        x_2, y_2 = landmarkList[8][1], landmarkList[8][2]       # tips of index
        x_3, y_3 = landmarkList[12][1], landmarkList[12][2]     # tips of middle
        x_4, y_4 = landmarkList[16][1], landmarkList[16][2]     # tips of ring
        x_5, y_5 = landmarkList[20][1], landmarkList[20][2]     # tips of pinky

        cv2.circle(frame, (x_0, y_0), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_3, y_3), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_4, y_4), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_5, y_5), 7, (0, 255, 0), cv2.FILLED)

        TI = hypot(x_2-x_1, y_2-y_1)        # thumb and index
        interTI = np.interp(TI, [15,220], [0,100])

        MP = hypot(x_3-x_0, y_3-y_0)        # middle and palm
        interMP = np.interp(MP, [15,300], [0,100])
        RP = hypot(x_4-x_0, y_4-y_0)        # ring and palm
        interRP = np.interp(RP, [15,300], [0,100])
        PP = hypot(x_5-x_0, y_5-y_0)        # pinky and palm
        interPP = np.interp(PP, [15,300], [0,100])

        if int(interMP) < 55 and int(interRP) < 45 and int(interPP) < 45:
                    cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)
                    volume = "set volume output volume " + str(int(interTI))
                    osascript.osascript(volume)


    # Display frame
    FRAME_WINDOW.image(frame)

# Release the webcam when done
camera.release()