# Finger Paint

This web app uses opencv as well as mediapipe to place landmarks on a users hand allowing them to track the finger tip movement and draw on a digital canvas.

<img width="1680" alt="Screenshot 2025-05-12 at 3 49 35 AM" src="https://github.com/user-attachments/assets/865e9ea0-911f-4688-a73a-9a16de9611bc" />

## Running
1. Choose your video capture device
2. Run this in the root directory of the project
```
uvicorn main:app --reload
```

## How to

1. Your index finger is the pen tip.
2. Tap your thumb and middle finger to toggle drawing on/off.
3. Tap your thumb and index finger to clear the canvas.
4. Your whole hand has to be present.

## What next?
1. Add an eraser
2. Add more colors
3. Save image
4. Multihand drawing
