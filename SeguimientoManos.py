import math
import cv2
import mediapipe as mp
import time


class HandsDetector:
    #-------------------Inicializamos los parametros de la deteccion----------------
    def __init__(self, mode=False, max_hands=2, model_complexity=1, conf_detection=0.5, conf_follow=0.5):
        self.list = None
        self.results = None
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = model_complexity
        self.conf_detection = conf_detection
        self.conf_follow = conf_follow

        # ---------------------------- Creamos los objetos que detectaran las manos y las dibujaran----------------------
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.conf_detection,
                                         self.conf_follow)
        self.sketch = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    #----------------------------------------Funcion para encontrar las manos-----------------------------------
    def find_hands(self, frame, draw=True):
        img_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_color)

        if self.results.multi_hand_landmarks:
            for mano in self.results.multi_hand_landmarks:
                if draw:
                    self.sketch.draw_landmarks(frame, mano, self.mp_hands.HAND_CONNECTIONS)
        return frame

    #------------------------------------Funcion para encontrar la posicion----------------------------------
    def find_position(self, frame, hand_number=0, draw=True, color=[]):
        x_list = []
        y_list = []
        bbox = []
        player = 0
        self.list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            test = self.results.multi_hand_landmarks
            player = len(test)
            #print(player)
            for index, lm in enumerate(my_hand.landmark):
                height, width, c = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                x_list.append(cx)
                y_list.append(cy)
                self.list.append([index, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 0), cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bbox = x_min, y_min, x_max, y_max
            if draw:
                cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), color, 2)
        return self.list, bbox, player

    #----------------------------------Funcion para detectar y dibujar los dedos arriba------------------------
    def fingers_up(self):
        fingers = []
        if self.list[self.tip[0]][1] > self.list[self.tip[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for i in range(1, 5):
            if self.list[self.tip[i]][2] < self.list[self.tip[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    #--------------------------- Funcion para detectar la distancia entre dedos----------------------------
    def distance(self, p1, p2, frame, draw=True, r=15, t=3):
        x1, y1 = self.list[p1][1:]
        x2, y2 = self.list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]


#----------------------------------------------- Funcion principal-------------------- ----------------------------
def main():
    p_time = 0
    c_time = 0

    cap = cv2.VideoCapture(0)
    detector = HandsDetector()

    while True:
        ret, frame = cap.read()
        frame = detector.find_hands(frame)
        m_list, bbox, _ = detector.find_position(frame)
        #if len(m_list) != 0:
        #print(m_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Manos", frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
