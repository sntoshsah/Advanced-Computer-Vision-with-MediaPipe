import cv2 as cv
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode=False, max_no_hands = 2, detectConf = 0.5 ,trackConf = 0.5):
        self.mode = mode
        self.maxHands = max_no_hands
        self.detectConf = detectConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector(detectConf=1)

    while True:
        success, frame = cap.read()
        # Check if frame was read correctly
        if not success:
            print("Can't receive frame (stream end?). Exiting...")
            break

        frame = detector.findHands(frame)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
        cv.imshow('Video', frame)

        # Exit when 'q' key is pressed
        if cv.waitKey(1) == ord('q'):
            break

# Release resources
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()


