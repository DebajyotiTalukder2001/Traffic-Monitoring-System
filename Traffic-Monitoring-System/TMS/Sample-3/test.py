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

cap = cv2.VideoCapture('Videos/vid3.mp4')

my_file = open("coco.txt", "r")  # Class File
data = my_file.read()
class_list = data.split("\n")
# print(class_list)


count = 0

# Co-ordinates of the desired region (Region of Interest or ROI).
# above co-ordinates can vary according to the input video-footage or test cases.
# So, we have to put proper co-ordinates using the mouse co-ordinate.

area = [(188, 291), (670, 291), (698, 319), (148, 319)]
area2 = [(95, 363), (744, 363), (854, 441), (2, 441)]

area_c = set()  # Initialize empty Set
tracker = Tracker()  # Initialize the Tracker object. (Defined in tracker file)

#speed_limit = 60
speed_limit = 120

vehicles_entering = {}  # Initialize empty dictionary
vehicles_elapsed_time = {}  # Initialize empty dictionary
vehicles_entering_backward = {}  # Initialize empty dictionary

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
    
    # a = results[0].xyxy[0] 
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
        elif 'motorcycle' in c:
            list.append([x1, y1, x2, y2])
        elif 'truck' in c:
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

       

        # Area-1 (forward moving vehicles enter here first)

        if results >= 0:
                
            #Uncomment the below parts if the video contains vehicles coming from both direction

            # # forward vehicles
            # if id not in vehicles_entering_backward:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                     0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                Init_time = time.time()

                if id not in vehicles_entering:
                    vehicles_entering[id] = Init_time
                else:

                    Init_time = vehicles_entering[id]

            # # backward vehicles
            # else:
            #     try:

            #         elapsed_time = time.time() - vehicles_entering_backward[id]

            #     except KeyError:
            #             pass
            #     cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            #     cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
            #                     0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #     cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                    

            #     if id not in vehicles_elapsed_time:
            #                 vehicles_elapsed_time[id] = elapsed_time
            #     else:
            #             try:
            #                    # Speed -> distance/elapsed time

            #                     elapsed_time = vehicles_elapsed_time[id]

            #                     dist = 30  # Distance between two region

            #                     speed_KH = (dist/elapsed_time)*3.6

            #                     cv2.putText(frame, str(int(speed_KH))+'Km/h', (x4, y4),
            #                                 cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            #                     #    cv2.putText(frame, str(elapsed_time), (x4, y4),
            #                     #                 cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            #             except ZeroDivisionError:
            #                     pass

            #             if speed_KH >= speed_limit:
            #                    # Display a warning message
            #                     cv2.waitKey(500)
            #                     cv2.putText(frame, "Speed limit violated!", (440, 112),
            #                                 cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            #                     cv2.putText(frame, 'Detected', (cx, cy),
            #                                 cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            #                     cv2.waitKey(500)

        # Area-2 | Main Area (backward moving vehicles enter here first)


        if results2 >= 0:

             #Uncomment the below parts if the video contains vehicles coming from both direction
             # # backward vehicles
            if id not in vehicles_entering:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                     0.8, (0, 255, 255), 2, cv2.LINE_AA)
                area_c.add(id)  # Vehicle-counter
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                # Init_time = time.time()

                # if id not in vehicles_entering_backward:
                #     vehicles_entering_backward[id] = Init_time
                # else:

                #     Init_time = vehicles_entering_backward[id]

             # # forward vehicles
            else:
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
                             # Speed -> distance/elapsed time

                        elapsed_time = vehicles_elapsed_time[id]

                        dist = 30  # Distance between two region

                        speed_KH = (dist/elapsed_time)*3.6

                        cv2.putText(frame, str(int(speed_KH))+'Km/h', (x4, y4),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                        #    cv2.putText(frame, str(elapsed_time), (x4, y4),
                        #                 cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                    except ZeroDivisionError:
                        pass

                    if speed_KH >= speed_limit:
                             # Display a warning message
                        cv2.waitKey(500)
                        cv2.putText(frame, "Speed limit violated!", (315, 90),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, 'Detected', (cx, cy),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.waitKey(500)

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

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2) # Area-1
    cv2.polylines(frame, [np.array(area2, np.int32)],
                  True, (0, 255, 0), 2)  # Area-2 | Main Area
    cnt = len(area_c)
    cv2.putText(frame, ('Vehicle-Count:-')+str(cnt), (452, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("TMS", frame)
    # put 0 to freeze the frame
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
