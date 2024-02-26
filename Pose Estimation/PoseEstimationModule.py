import cv2 as cv
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.results = None
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
                                      self.min_tracking_confidence, )

    '''
    def __init__(self, mode=False, upbody = False, smooth=True,
                 detectionCon = 0.5,
                 trackCon = 0.5):
        self.mode = mode,
        self.upbody = upbody,
        self.smooth = smooth,
        self.detectionCon = detectionCon,
        self.trackCon = trackCon
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upbody, self.smooth, self.detectionCon, self.trackCon)
'''

    def find_pose(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                return img

    def find_position(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for Id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print('Id:{Id}, landmark: {lm}'.format(Id, lm))
                # print(Id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 3, [255, 0, 0], -1)
            return lmlist


def main():
    ptime = 0
    ctime = 0
    detector = PoseDetector()
    cap = cv.VideoCapture('50 Must-know BEGINNER YOGA POSES _ Yoga for beginners.mp4')

    while True:
        ret, frame = cap.read()

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        img = detector.find_pose(frame)
        lmlist = detector.find_position(frame)
        print(lmlist)
        cv.putText(img, str(int(fps)), [500, 500], cv.FONT_HERSHEY_SIMPLEX, 1, [255, 30, 30], 2)
        cv.imshow('Frame', frame)

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
