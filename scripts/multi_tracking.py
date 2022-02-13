# Usage
# python scripts/multi_tracking.py -v 'resources/group_of_people_05.mp4' -t 'csrt'

import cv2
import sys
import argparse
from random import randint
import datetime

#########################################################################################################
#################################### Select Tracking Algorithm ##########################################
#########################################################################################################

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,  default='resources/group_of_people_01.mp4', help="Path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
args = vars(ap.parse_args())

def create_tracker_by_name(tracker_type):
    if tracker_type == tracker_types[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('[ERROR] Invalid selection! Available tracker: ')
        for t in tracker_types:
            print(t.lower())

    return tracker

print('[INFO] selected tracker: ' + str(args["tracker"].upper()))


#########################################################################################################
############################################# Load Video ################################################
#########################################################################################################

video = cv2.VideoCapture(args["video"])
# load video
if not video.isOpened():
    print('[ERROR] video file not loaded')
    sys.exit()
ok, frame = video.read()
if not ok:
    print('[ERROR] no frame captured')
    sys.exit()

print('[INFO] video loaded and frame capture started')


#########################################################################################################
###################################### Select Objects to track ##########################################
#########################################################################################################

# Define list for bounding boxes
# and tracking rectangle colour
bboxes = []
colours = []

while True:
    # Open selector and wait for selected ROIs
    bbox = cv2.selectROI('MultiTracker', frame)
    print('[INFO] select ROI')
    print('[INFO] press SPACE or ENTER to confirm selection')
    print('[INFO] press q to exit selection or any other key to continue')
    # Add ROIs to list of bounding boxes
    bboxes.append(bbox)
    # Create random colour for each box
    colours.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    # Wait until user presses q to quit selection
    k = cv2.waitKey(0) & 0xff
    if k == ord('q'):
        break


#########################################################################################################
####################################### Initialize ROI Tracking #########################################
#########################################################################################################


multi_tracker = cv2.legacy.MultiTracker_create()

for bbox in bboxes:
    multi_tracker.add(create_tracker_by_name(args["tracker"].upper()), frame, bbox)

while video.isOpened():
    # get frames from video
    ok, frame = video.read()
    if not ok:
        print('[INFO] end of video file reached')
        break

    # get new bounding box coordinates
    # for each frame from tracker
    ok, boxes = multi_tracker.update(frame)
    if not ok:
        cv2.putText(frame, 'Track Loss', (10, 50), cv2.QT_FONT_NORMAL, 1, (0, 0, 255))
    # use coordinates to draw rectangle
    for i, new_box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in new_box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), colours[i], 3)

    cv2.putText(frame, str(args["tracker"].upper()), (10, 30), cv2.QT_FONT_NORMAL, 1, (255, 255, 255))
    cv2.imshow("MultiTracker", frame)
    # press 'q' to break loop and close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()