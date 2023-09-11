# /////////////////////////////Tracker Module/////////////////////////////////////////////////////////////////////////////\

#Centroid Tracking algorithm
#Eucledian Distance Tracker

import math


class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        #The self.center_points variable is a dictionary that stores the center positions of the objects that have been detected. 
        #The key of the dictionary is the ID of the object, and the value is the tuple of the x and y coordinates of the object's center point.
        self.center_points = {}
        # Keep the count of the IDs
        #The self.id_count variable is a counter that keeps track of the ID of the next object that is detected.
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        #The objects_bbs_ids variable is a list that will store the bounding boxes and IDs of the objects that have been detected in the current frame.
        objects_bbs_ids = []

        # Get center point of new object
        #The for loop iterates over the list of objects, objects_rect. 
        #For each object, the code calculates the center point of the object by taking the average of the x and y coordinates of the top-left and bottom-right corners of the object's bounding box.
        #The center point is then assigned to the variable cx and cy.

        #The objects_bbs_ids variable is initialized to an empty list. 
        #• The for loop iterates over the list of objects, objects_rect. 
        #• The rect variable is a tuple that contains the x, y, w, and h coordinates of the current object's bounding box. 
        #• The x, y, w, and h variables are assigned the values from the rect tuple. 
        #• The cx variable is assigned the average of the x and w coordinates.
        #• The cy variable is assigned the average of the y and h coordinates. 
        #• The (cx, cy) tuple is appended to the objects_bbs_ids list.

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            #The same_object_detected variable is a boolean variable that is used to keep track of whether or not the current object has been detected before. 
            #The for loop iterates over the dictionary self.center_points. 
            #For each object in the dictionary, the code calculates the distance between the center point of the current object and the center point of the object in the dictionary.
            #If the distance is less than a threshold value (50), then the code concludes that the same object has been detected and the same_object_detected variable is set to True. 
            #The object's bounding box and ID are then appended to the objects_bbs_ids list.
            #The Threshold value can vary according to input video or test-cases.
            
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 80:
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object 
            ##The same_object_detected variable is a boolean variable that is used to keep track of whether or not the current object has been detected before. 
            #The if statement checks if the same_object_detected variable is False. 
            #If it is, then the code assigns a new ID to the object and appends the object's bounding box and ID to the objects_bbs_ids list.

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        #The new_center_points dictionary is a new dictionary that will be used to store the center points of the objects that have been detected in the current frame. 
        #The for loop iterates over the list of objects, objects_bbs_ids. 
        #For each object, the code retrieves the object's ID and center point from the self.center_points dictionary. 
        #The center point is then added to the new_center_points dictionary.

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        #The self.center_points dictionary is updated with the new_center_points dictionary. 
        #The new_center_points dictionary contains the center points of the objects that have been detected in the current frame.
        #The return statement returns the list of objects, objects_bbs_ids.

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# /////////////////////////////Tracker Module/////////////////////////////////////////////////////////////////////////////

# /////////////////////////////Speed Detector Module/////////////////////////////////////////////////////////////////////////////
import numpy as np

class SpeedEstimator:
    def __init__(self,posList,fps):
        self.x=posList[0]
        self.y=posList[1]
        self.fps=fps
        
    def estimateSpeed(self):
        # Distance / Time -> Speed
        d_pixels=math.sqrt(self.x **2 + self.y**2)

        # ppm values can vary
        # For example, the horizontal resolution of a camera is 1920x1080. 
        # In other words, it shows the image that appears on the screen with 1920 pixels horizontally and 1080 pixels vertically. 
        # If the horizontal width displayed on the screen is 192 meters when measured in meters, it is 1920 pixels/192 meters = 10 pixels/meter = 10PPM.
        # In other words, 10 pixels correspond to 1 meter.
        # ppm = 8
        ppm = 15
        d_meters=int(d_pixels*ppm)
        speed=d_meters/self.fps*3.6
        speedInKM=np.average(speed)
        return int(speedInKM)
# /////////////////////////////Speed Detector Module/////////////////////////////////////////////////////////////////////////////