import cv2
from ultralytics import YOLO

import HandsDetector as hd


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

hand_detector = hd.HandsDetector(conf_detection=0.9)
model = YOLO('best.pt')


while True:
    ret, frame = cap.read()

    frame = hand_detector.find_hands(frame, draw=False)

    list1, bbox, hand = hand_detector.find_position(frame, hand_number=0, draw=False, color=[0, 255, 0])

    if hand == 1:
        x_min, y_min, x_max, y_max = bbox

        x_min = x_min - 60
        x_max = x_max + 60
        y_min = y_min - 60
        y_max = y_max + 60

        cuted_image = frame[y_min:y_max, x_min:x_max]

        cuted_image = cv2.resize(cuted_image, (640, 640), interpolation=cv2.INTER_CUBIC)

        result = model.predict(cuted_image, conf=0.55)

        if len(result) != 0:

            for res in result:
                masks = res.masks
                coords = masks

                annotations = result[0].plot()

        cv2.imshow('Predict', annotations)

    cv2.imshow('Global', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
