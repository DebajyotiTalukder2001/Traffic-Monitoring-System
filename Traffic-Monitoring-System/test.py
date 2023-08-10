# pip install ultralytics --target=D:\System\Conda\Lib\site-packages

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
from math import dist
from tracker import*
model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('Videos/vid1.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0
speed = {}
area = [(225, 335), (803, 335), (962, 408), (57, 408)]
area_c = set()
tracker = Tracker()
speed_limit = 62

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
 #   print(results)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
#    print(px)
    list = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3+x4)//2
        cy = int(y3+y4)//2
        results = cv2.pointPolygonTest(
            np.array(area, np.int32), ((cx, cy)), False)

        if results >= 0:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            area_c.add(id)
            now = time.time()
            if id not in speed:
                speed[id] = now
            else:
                try:
                    prev_time = speed[id]
                    speed[id] = now
                    dist = 5
                    a_speed_ms = dist / (now - prev_time)
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.putText(frame, str(int(a_speed_kh))+'Km/h', (x4, y4),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    speed[id] = now
                except ZeroDivisionError:
                    pass

                # Check if the speed exceeds the speed limit
                if a_speed_kh >= speed_limit:
                    # Display a warning message
                    cv2.putText(frame, "Speed limit violated!", (440, 115),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                    # Display the message for 3 seconds
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        cv2.imshow("RGB", frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    cnt = len(area_c)
    cv2.putText(frame, ('Vehicle-Count:-')+str(cnt), (452, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
