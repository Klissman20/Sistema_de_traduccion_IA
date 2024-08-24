import cv2
import os
import time

import HandsDetector as sm

HTTP = 'http://'
IP_ADDRESS = '192.168.1.176'
URL =  HTTP + IP_ADDRESS + ':4747/mjpegfeed?640x480'

folder_name = 'D'
path = r'C:\Users\hp\Universidad EIA\Klissman Esteban Idarraga Gomez - Sistema de traduccion automatico\data'
id = '003'

# Crear la estructura de carpetas
folder = os.path.join(path, folder_name, id)
folder_der = os.path.join(folder, 'der')
folder_izq = os.path.join(folder, 'izq')

# Crear subcarpetas para las imágenes completas y con hand detection
folders = [os.path.join(folder_der, 'frame_completo'),
           os.path.join(folder_der, 'hand_detection'),
           os.path.join(folder_izq, 'frame_completo'),
           os.path.join(folder_izq, 'hand_detection')]

for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

cap = cv2.VideoCapture(1)
#cap.set(3, 1280)
#cap.set(4, 720)

hand_detector = sm.HandsDetector(conf_detection=0.9)

count = 0  # Definir 'count' fuera de la función para que sea accesible globalmente

def capturar_imagenes(lado, num_fotos):
    global count  # Usar 'count' como una variable global
    for i in range(num_fotos):
        # Capturar imagen completa
        ret, frame = cap.read()
        cv2.imshow('Frame Completo', frame)
        cv2.imwrite(os.path.join(folders[0 if lado == 'der' else 2], 
                                 f'{folder_name}{id}_{count}_{"r" if lado == "der" else "l"}_f.jpg'), frame)
        
        # # Aplicar hand detection y capturar imagen
        # frame_hd = hand_detector.find_hands(frame, draw=False)
        # lista1, bbox, mano = hand_detector.find_position(frame_hd, hand_number=0, draw=False)
        
        # if mano == 1:
        #     x_min, y_min, x_max, y_max = bbox
        #     margin = 80
        #     recorted_frame = frame_hd[y_min - margin:y_max + margin, x_min-margin:x_max + margin]
        #     recorted_frame = cv2.resize(recorted_frame, (640, 640), interpolation=cv2.INTER_AREA)
        #     #cv2.imshow('Hand Detection', recorted_frame)
        #     cv2.imwrite(os.path.join(folders[1 if lado == 'der' else 3], 
        #                              f'{folder_name}{id}_{count}_{"r" if lado == "der" else "l"}_hd.jpg'), recorted_frame)
        
        count += 1  # Incrementar 'count' después de cada captura
        
        # Esperar brevemente entre capturas
        cv2.waitKey(100)

# Mostrar video en tiempo real para preparar la posición
print("Ajusta tu posición y presiona 's' para comenzar la captura...")
while True:
    ret, frame = cap.read()
    cv2.imshow('Preparación', frame)

    # Esperar a que el usuario presione 's' para iniciar la captura
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cv2.destroyWindow('Preparación')

# Capturar 100 imágenes de la mano derecha
print(f'Se tomará el registro de la letra {folder_name} de la mano derecha en 3 segundos...')
time.sleep(3)
capturar_imagenes('der', num_fotos=100)

# Mostrar video en tiempo real para preparar la mano izquierda
print("Prepárate para la captura de la mano izquierda. Presiona 's' para comenzar...")
while True:
    ret, frame = cap.read()
    cv2.imshow('Preparación', frame)

    # Esperar a que el usuario presione 's' para iniciar la captura
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cv2.destroyWindow('Preparación')

# Capturar 100 imágenes de la mano izquierda
print(f'Se tomará el registro de la letra {folder_name} de la mano izquierda en 3 segundos...')
time.sleep(3)
capturar_imagenes('izq', num_fotos=100)

cap.release()
cv2.destroyAllWindows()
