'''import cv2 as cv
import mediapipe as mp

class PoseDetector:
    def __init__(self, mode=False, upper_body=False, smooth=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.smooth_landmarks = smooth
        self.upper_body = upper_body
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upper_body, self.smooth_landmarks,
                                      self.min_detection_confidence, self.min_tracking_confidence)

    def find_pose(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 3, (255, 0, 0), -1)
        return lmlist

def main():
    detector = PoseDetector()
    cap = cv.VideoCapture('50 Must-know BEGINNER YOGA POSES _ Yoga for beginners.mp4')

    ptime = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        img = detector.find_pose(frame)
        lmList = detector.find_position(frame)
        print(lmList)

        cv.putText(img, str(int(fps)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 30, 30), 2)
        cv.imshow('Frame', img)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
'''

import cv2
import mediapipe as mp

class pose_detector():
    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode,
                                    self.model_complexity,
                                    self.smooth_landmarks,
                                    self.enable_segmentation,
                                    self.smooth_segmentation,
                                    self.min_detection_confidence,
                                    self.min_tracking_confidence,)

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img


    def get_landmark(self, img, draw=True):

        cords = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)
                cords.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return cords


def main():
    cap = cv2.VideoCapture(0)
    detector_pose = pose_detector()

    while True:
        success, img = cap.read()

        detector_pose.find_pose(img)

        cv2.imshow("Human Detector", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()