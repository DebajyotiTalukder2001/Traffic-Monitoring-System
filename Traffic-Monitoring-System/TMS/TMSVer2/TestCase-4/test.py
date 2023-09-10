# pip install ultralytics --target=D:\System\Conda\Lib\site-packages

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
from math import dist
from tracker import*
model = YOLO('yolov8s.pt')  # Load Pretrained Model

# get the mouse coordinates


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('TMS')

cv2.setMouseCallback('TMS', RGB)

# read Video

cap = cv2.VideoCapture('Videos/vid2.mp4')

my_file = open("coco.txt", "r")  # Class File
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

# Co-ordinates of the desired region (Region of Interest or ROI).
# above co-ordinates can vary according to the input video-footage or test cases.
# So, we have to put proper co-ordinates using the mouse co-ordinate.

area = [(170, 219), (1018, 219), (970, 286), (110, 286)]
area2 = [(85, 316), (1014, 316), (928, 392), (1, 392)]


area_c = set()  # Initialize empty Set

tracker = Tracker()  # Initialize the Tracker object. (Defined in the tracker file)

# speed_limit = 60
speed_limit = 80

vehicles_entering = {}  # Initialize empty dictionary
vehicles_elapsed_time = {}  # Initialize empty dictionary

while True:
    ret, frame = cap.read()
    # Check if frame is read
    if not ret:
        break
    count += 1
    # Skip Frames
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

   # The YOLO model is used to predict the bounding boxes of the objects in the frame.
   # The px variable is a Pandas DataFrame that stores the bounding boxes in a format that is easy to iterate over.

    results = model.predict(frame)

    # print(results)

    a = results[0].boxes.boxes

    px = pd.DataFrame(a).astype("float")
    # print(px)

    list = []  # Initialize empty List

    # The for loop iterates over the rows of the px DataFrame.
    # For each row, the following steps are performed:
    # 1. The x1, y1, x2, and y2 coordinates of the bounding box are extracted.
    # 2. The class ID of the object is extracted.
    # 3. The class name is obtained from the class_list variable.
    # 4. If the class name is "car", then the bounding box is added to the list variable.

    for index, row in px.iterrows():
        # print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
        if 'motorcycle' in c:
            list.append([x1, y1, x2, y2])
        elif 'truck' in c:
            list.append([x1, y1, x2, y2])
        elif 'bus' in c:
            list.append([x1, y1, x2, y2])

    # This line calls the update() method of the Tracker object.
    # The update() method takes a list of bounding boxes as input and returns a list of bounding boxes with the IDs of the tracked objects.

    bbox_id = tracker.update(list)

    # This line iterates over the list of bounding boxes returned by the update() method.
    # For each bounding box, the following steps are performed:
    # 1. The x3, y3, x4, and y4 coordinates of the bounding box are extracted.
    # 2. The ID of the object is extracted.
    # 3. The center coordinates of the bounding box are calculated.
    # 4. The pointPolygonTest(contour,pt,measureDist) function is used to check if the center coordinates of the bounding box are inside the specified area.

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3+x4)//2
        cy = int(y3+y4)//2

        results = cv2.pointPolygonTest(
            np.array(area, np.int32), ((cx, cy)), False)   

        results2 = cv2.pointPolygonTest(
            np.array(area2, np.int32), ((cx, cy)), False) 

        # Detection will be done within the region only
        # This line checks if the center coordinates of the bounding box are inside the specified area.
        # If they are, then the following steps are performed:
        # 1. A circle is drawn at the center of the bounding box.
        # 2. The ID of the object is displayed at the top-left corner of the bounding box.
        # 3. The bounding box is drawn in red.
        # 4. The ID of the object is added to the set area_c.
        # when, the vehicle passes through the Two regions, the elapsed time will be calculated.


        # Remember -- This method can detect vehicle speed in only one-direction at a time.

        # Area-1
        # rename it to results2 (if results2 >= 0) when considering backward direction of the vehicle.

        if results >= 0:

            # use this to show bounding box to check if working properly, otherwise not necessary.
            # cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            # cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
            #             0.8, (0, 255, 255), 2, cv2.LINE_AA)

            Init_time = time.time()

            if id not in vehicles_elapsed_time:
                vehicles_entering[id] = Init_time
            else:

                Init_time = vehicles_entering[id]



        # Area-2 | Main Area

        # rename it to results (if results >= 0) when considering backward direction of the vehicle.

        if results2 >= 0:
            try:
                elapsed_time = time.time() - vehicles_entering[id]
            except KeyError:
                pass
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            area_c.add(id)  # Vehicle-counter

            if id not in vehicles_elapsed_time:
                vehicles_elapsed_time[id] = elapsed_time
            else:
                try:
                    #Speed -> distance/elapsed time

                    elapsed_time = vehicles_elapsed_time[id]

                    dist = 12 #Distance between two region
                    
                    speed_KH = (dist/elapsed_time)*3.6

                    cv2.putText(frame, str(int(speed_KH))+'Km/h', (x4, y4),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                #    cv2.putText(frame, str(elapsed_time), (x4, y4),
                #                 cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                except ZeroDivisionError:
                    pass

                if speed_KH >= speed_limit:
                    # Display a warning message
                    cv2.waitKey(300)
                    cv2.putText(frame, "Speed limit violated!", (440, 112),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.waitKey(300)

    # This code draws the specified area on the frame, displays the number of vehicles in the area, and releases the video capture object and destroys all windows.
    # The following steps are performed:
    # 1. The polylines(image,pts,isClosed,color,thickness) function is used to draw the specified area on the frame.
    # The area variable is a list of points that define the area.
    # 2. The len() function is used to get the number of vehicles in the area.
    # 3. The putText(image, text, org, font, fontScale, color, thickness, lineType) function is used to display the number of vehicles in the area on the frame.
    # 4. The imshow() function is used to display the frame.
    # 5. The waitKey() function waits for a key press.
    # If the user presses the Esc key, then the loop breaks.
    # 6. The release() function releases the video capture object.
    # 7. The destroyAllWindows() function destroys all windows.

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2) #Area-1
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2) #Area-2 | Main Area

    cnt = len(area_c)
    cv2.putText(frame, ('Vehicle-Count:-')+str(cnt), (452, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("TMS", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()