import cv2 as cv
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

ptime = 0
ctime = 0
cap = cv.VideoCapture('50 Must-know BEGINNER YOGA POSES _ Yoga for beginners.mp4')

while True:
    ret, frame = cap.read()
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for Id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img_rgb.shape
            # print('Id:{Id}, landmark: {lm}'.format(Id, lm))
            print(Id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv.circle(frame,(cx, cy), 3, [255, 0, 0], -1 )


    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv.putText(frame, str(int(fps)), [500, 500], cv.FONT_HERSHEY_SIMPLEX, 1, [255,30,30], 2)
    cv.imshow('Frame', frame)


    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()