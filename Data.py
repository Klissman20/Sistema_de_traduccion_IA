import cv2
import os

import HandsDetector as sm

folder_name = 'A'
path = r'C:\Users\hp\Universidad EIA\Klissman Esteban Idarraga Gomez - Sistema de traduccion automatico\data'
folder = path + '/' + folder_name

if not os.path.exists(folder):
    os.mkdir(folder)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

count = 0

hand_detector = sm.HandsDetector(conf_detection=0.9)

while True:
    ret, frame = cap.read()

    frame = hand_detector.find_hands(frame, draw=True)
    
    lista1, bbox, mano = hand_detector.find_position(frame, hand_number=0, draw=False)
    
    if mano == 1:
        x_min, y_min, x_max, y_max = bbox
        
        margin = 80
        
        # cv2.rectangle(frame, (x_min - margin, y_min - margin), (x_max + margin, y_max + margin), (0, 255, 0), 2)
        
        recorted_frame = frame[y_min - margin:y_max + margin, x_min-margin:x_max + margin]
        
        cv2.imshow('Frame', recorted_frame)
        
        recorted_frame = cv2.resize(recorted_frame, (640,640), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(folder + '/A{}.jpg'.format(count), recorted_frame)
        
        count += 1
        

    #cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count == 10:
        break
