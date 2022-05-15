# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

previous_frames = []

def get_reference_frame(previous_frames):
    n = len(previous_frames)
    if n == 0:
        return None
    ref_frame = previous_frames[0]
    for i in range(len(previous_frames)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            ref_frame = cv2.addWeighted(previous_frames[i], alpha, ref_frame, beta, 0.0)
    return ref_frame

def update_previous_frames(previous_frames, new_frame):
    n = len(previous_frames)
    if n < previous_tracked_frames_num:
        previous_frames.append(new_frame)
    else:
        previous_frames.pop(0)
        previous_frames.append(new_frame)


def string_parser(cords_str):
    cords = []
    temp = cords_str.split(";")
    print(temp)
    rectangle = []
    for i in temp:
        temp2 = i.split(",")
        print(temp2)
        rectangle.append(int(temp2[0][1:]))
        rectangle.append(int(temp2[1][:-1]))
        if len(rectangle) == 4:
            cords.append(rectangle)
            print(rectangle)
            rectangle = []
        #print(cords)
    return cords


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-d", "--debug", help="enabling debug mode", action='store_true')
ap.add_argument("-m", "--mask", help="mask regions coords in format (lb1x,lb1y);(rt1x,rt1y);(lb2x,lb2y);(rt2x,rt2y)")
ap.add_argument("-b", "--bottom_threshold", help="bottom frame delta threshold", type=int, default=25)
ap.add_argument("-p", "--previous_frames", help="previous frames used to create reference frame", type=int, default=0)

args = ap.parse_args()
args_dict = vars(args)

if args_dict.get("mask", None) is None:

    mask_coordinates = [[0,0,500,500]]
else:
    mask_coordinates = string_parser(args_dict["mask"])

if args_dict.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args_dict["video"])

b_threshold = args_dict["bottom_threshold"]
previous_tracked_frames_num = args_dict["previous_frames"]

ref_frame = None
check = False

while True:
    frame = vs.read()
    frame = frame if args_dict.get("video", None) is None else frame[1]
    text = "No movement"

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    if len(mask_coordinates) == 0:
        frame_dimensions = frame.shape
        mask_coordinates.append([0,0,frame_dimensions[1], frame_dimensions[0]])
    for mask in mask_coordinates:
        cv2.rectangle(frame, (mask[0],mask[1]), (mask[2], mask[3]), (225, 0, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if ref_frame is None:
        ref_frame = gray
    elif previous_tracked_frames_num > 0:
        ref_frame = get_reference_frame(previous_frames)
        update_previous_frames(previous_frames, gray)

    frameDelta = cv2.absdiff(ref_frame, gray)

    thresh = cv2.threshold(frameDelta, b_threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    for mask in mask_coordinates:
        part = thresh[mask[0]:mask[2], mask[1]:mask[3]]
        if args.debug:
            cv2.imshow("Part" + str(mask[0]), part)
        cnts = cv2.findContours(part.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            if cv2.contourArea(c) < args_dict["min_area"]:
                continue
            check = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x+mask[0], y+mask[1]), (x+mask[0] + w, y + mask[1] + h), (0, 255, 0), 2)
    if check:
        text = "Movement detected"
    cv2.putText(frame, "Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("Security Feed", frame)
    check = False
    if args.debug:
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("b"):
        b_threshold += 2
        print("bottom threshold set to: ", b_threshold)
    elif key == ord("v"):
        if b_threshold >= 2:
            b_threshold -= 2
        print("bottom threshold set to: ", b_threshold)

vs.stop() if args_dict.get("video", None) is None else vs.release()
cv2.destroyAllWindows()

