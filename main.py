# import cv2 as cv
# import numpy as np
# cap = cv.VideoCapture('exp.mp4')
# while cap.isOpened():
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     cv.imshow('frame', frame)
#     if cv.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv.destroyAllWindows()

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np


def string_parser(cords_str):
    cords = []
    temp = cords_str.split(";")
    #print(temp)
    '''for i in range (0,len(temp),2):
        temp2 = temp[i].split(",")
        temp2.append(temp[i+1].split(","))
        print(temp2)
        cords.append([temp2[0][1:],temp2[2][1:],temp2[1][:len(temp2[1])-2],temp2[3][len(temp2[3])-2]]) #x1,y1,x2,y2
        print(cords)'''
    for i in temp:
        temp2 = i.split(",")
        cords.append([int(temp2[0][1:]),int(temp2[1]),int(temp2[2]),int(temp2[3][:len(temp2[3])-1])])
        #print(cords)
    return cords

#def find_movement():

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
# Debug mode
ap.add_argument("-d", "--debug", help="enabling debug mode", action='store_true')
#

# Setting mask
ap.add_argument("-m", "--mask", help="mask regions coordinates in format (lb1x,rt1y);(lb2x,rt2y)")
#

# Setting bottom frame delta threshold
ap.add_argument("-b", "--bottom_threshold", help="bottom frame delta threshold", type=int, default=25)
#

# Setting top frame delta threshold
ap.add_argument("-t", "--top_threshold", help="top frame delta threshold", type=int, default=255)
#

args = ap.parse_args()
args_dict = vars(args)

'''mask_coordinates = string_parser(args_dict["mask"])
b_threshold = args_dict["bottom_threshold"]
t_threshold = args_dict["top_threshold"]'''
mask_coordinates = string_parser("(200,200,300,300);(20,20,150,150)")
b_threshold = 25
t_threshold = 225
parts = []
check = False
# if the video argument is None, then we are reading from webcam
if args_dict.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args_dict["video"])
# initialize the first frame in the video stream
firstFrame = None
# loop over the frames of the video


while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame = vs.read()
    frame = frame if args_dict.get("video", None) is None else frame[1]
    text = "No movement"
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    for mask in mask_coordinates:
        cv2.rectangle(frame, (mask[0],mask[1]), (mask[2], mask[3]), (225, 0, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue


    # compute the absolute difference between the current frame and
    # reference frame

    # foo = cv2.subtract(firstFrame, gray, dtype=cv2.CV_64F)
    # frameDelta = np.abs(foo, out=foo)
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, b_threshold, t_threshold, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    for mask in mask_coordinates:
    #part = thresh[mask_coordinates[0][0]:mask_coordinates[0][2], mask_coordinates[0][1]:mask_coordinates[0][3]]

        part = thresh[mask[0]:mask[2], mask[1]:mask[3]]
        cv2.imshow("Part" + str(mask[0]), part)
        cnts = cv2.findContours(part.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    # loop over the contours
        for c in cnts:
        # if the contour is too small, ignore it
            if cv2.contourArea(c) < args_dict["min_area"]:
                continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
            check = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the frame and record if the user presses a key
    if check:
        text = "Movement detected"
        # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    check = False
    if args.debug:
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
# cleanup the camera and close any open windows
vs.stop() if args_dict.get("video", None) is None else vs.release()
cv2.destroyAllWindows()

