import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, max_num_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_num_hands,1)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame

    def find_position(self, frame, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id , lm in enumerate(my_hand.landmark):
                h,w,c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                lm_list.append([id, cx, cy])
                if draw:
                    if id == 0:
                        cv.circle(frame, (cx, cy), 15, (255, 50, 0), -1)
        return lm_list

def main():
    p_time = 0
    c_time = 0

    cap = cv.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()

        if not success:
            print("Can't receive frame (stream end?). Exiting...")
            break

        frame = detector.find_hands(frame)
        position = detector.find_position(frame)
        if len(position) != 0:
            print(position)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
        cv.imshow('Hand Tracking', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
