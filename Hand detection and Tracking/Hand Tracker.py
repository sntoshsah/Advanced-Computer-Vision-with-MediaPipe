import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
while True:
    success, frame = cap.read()
    # Check if frame was read correctly
    if not success:
        print("Can't receive frame (stream end?). Exiting...")
        break

    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id , lm in enumerate(handLms.landmark):
                h,w,c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    cv.circle(frame, (cx,cy),15,(255,50,0), -1)
            mpDraw.draw_landmarks(frame, handLms,mpHands.HAND_CONNECTIONS)

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
