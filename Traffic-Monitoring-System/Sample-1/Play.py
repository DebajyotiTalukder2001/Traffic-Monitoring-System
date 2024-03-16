
import cv2
import pandas as pd
import numpy as np

#get the mouse coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('TMS')
cv2.setMouseCallback('TMS', RGB)

#read Video

cap = cv2.VideoCapture('Videos/vid1.mp4')


while True:
    ret, frame = cap.read()
    #Check if frame is read
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    cv2.imshow("TMS", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
