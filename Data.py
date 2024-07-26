import cv2
import os

import SeguimientoManos as sm

folder_name = 'A'
path = ''
folder = path + '/' + folder_name

if not os.path.exists(folder):
    os.mkdir(folder)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

hand_detector = sm.HandsDetector(conf_detection=0.9)

while True:
    ret, frame = cap.read()

    frame = hand_detector.find_hands(frame, draw=True)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
