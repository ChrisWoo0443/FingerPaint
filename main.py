import streamlit as st
import cv2
import mediapipe as mp
import numpy as np


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

    # Display frame
    FRAME_WINDOW.image(frame)

# Release the webcam when done
camera.release()