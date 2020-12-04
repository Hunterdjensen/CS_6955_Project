import cv2
import numpy as np
from numpy import genfromtxt
import copy
import imutils
from os import listdir
from os.path import isfile, join
import os
import re
from ColorDetectionTest5 import minEnclosingQuad
import csv

drawing = False  # true if mouse is pressed
making_new_point = True  # if True, it's a new point
ix, iy = -1, -1
points = []
points_history = []
point_idx = 0
frame_idx = 0
exit_flag = False
hsv_worked_flag = True


# 'optional' argument is required for trackbar creation parameters
def nothing(self):
    pass


def box_to_list(box):
    list_now = [[(box[0][0], box[0][1]), (box[1][0], box[1][1]), (box[2][0], box[2][1]), (box[3][0], box[3][1])]]
    return list_now


# Formats points in the order of a box: [[(25, 229), (28, 33), (168, 29), (168, 227)]]
# Bottom left, top left, top right, bottom right
def order_points(points):
    points_new = copy.deepcopy(points)
    for j, frame in enumerate(points):
        min_point = None
        min_total = 999999
        for i, point in enumerate(frame):
            if (point[0] + point[1]) < min_total:
                min_total = point[0] + point[1]
                min_point = i
        if min_point is None:
            exit("ERROR: There is no minumum point??")
        points_new[j][1] = frame[min_point]
        points_new[j][3] = frame[min_point-2]
        if frame[min_point-1][1] > frame[min_point-3][1]:
            points_new[j][0] = frame[min_point-1]
            points_new[j][2] = frame[min_point-3]
        else:   # Else flip the other two corners
            points_new[j][0] = frame[min_point-3]
            points_new[j][2] = frame[min_point-1]
    return points_new


def draw_lines():
    global img, orig_img
    img = orig_img.copy()
    for j, frame in enumerate(points):
        # for i in range(0, len(frame)):
        #     cv2.circle(img, points[j][i], 2, (0, 0, 255), -1)
        #     cv2.line(img, points[j][i], points[j][i - 1], (0, 0, 255), thickness=1)
        for i, point in enumerate(frame):
            cv2.circle(img, point, 1, (0, 0, 255), -1)
            cv2.line(img, point, points[j][i - 1], (0, 0, 255), thickness=1)


# mouse callback function
def mouse_movement(event, x, y, flags, param):
    global ix, iy, drawing, making_new_point, points, point_idx, frame_idx, img, orig_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        points_history.append(copy.deepcopy(points))
        making_new_point = True
        for j, frames in enumerate(points):
            for i, point in enumerate(frames):
                if (abs(x - point[0]) < 4) and (abs(y - point[1]) < 4):
                    # print("Modifying old point!")
                    making_new_point = False
                    point_idx = i
                    frame_idx = j
        if making_new_point:
            frame_idx = len(points) - 1
            if len(points) > 0:  # If there's at least one frame already
                if len(points[frame_idx]) >= 4:
                    points.append([])  # Create a new empty frame
                    frame_idx += 1
                point_idx = len(points[frame_idx])  # Not -1 because we're allocating for the new point
            else:
                points.append([])  # Create empty starting frame
                point_idx = 0
            points[frame_idx].append((0, 0))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            points[frame_idx][point_idx] = (x, y)
            draw_lines()

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points[frame_idx][point_idx] = (x, y)
        draw_lines()


# my_path = 'Python_Matching/raw/fronts/'
my_path = '../Examples/'
onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]
onlyfiles = sorted(onlyfiles)

os.chdir(my_path)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 765, 765)
cv2.setMouseCallback('image', mouse_movement)

outfile_name = 'classification_results.txt'
outfile = open(outfile_name, "a")
outfile.close()

# assign strings for ease of coding
hh = 'Hue High'
hl = 'Hue Low'
sh = 'Saturation High'
sl = 'Saturation Low'
vh = 'Value High'
vl = 'Value Low'
wnd = 'image'
# Begin Creating trackbars for each
cv2.createTrackbar(hl, wnd, 0, 179, nothing)
cv2.createTrackbar(hh, wnd, 0, 179, nothing)
cv2.createTrackbar(sl, wnd, 0, 255, nothing)
cv2.createTrackbar(sh, wnd, 0, 255, nothing)
cv2.createTrackbar(vl, wnd, 0, 255, nothing)
cv2.createTrackbar(vh, wnd, 0, 255, nothing)

hsv_key = genfromtxt('hsv_key.csv', delimiter=',', dtype=str)

# hsv_key = np.zeros([1, 7], dtype='object')
# hsv_key[0][0] = 'starter.jpg'
# hsv_key[0][2] = 179
# hsv_key[0][4] = 255
# hsv_key[0][6] = 255
# print(hsv_key)


def print_csv():
    with open('hsv_key.csv', 'w', newline='') as csvfile:
        csv_write = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i in range(hsv_key.shape[0]):
            csv_write.writerow(hsv_key[i])


# while (1):
for filename in onlyfiles:
    found = False
    # Search the entire outfile to see if this filename has already been processed
    with open(outfile_name) as f:
        if filename in f.read():
            found = True
    if not found and re.match(r".*.jpe", filename):
        # Initialize new photo
        img = cv2.imread(filename)
        # img = imutils.resize(img, height=255)
        img = cv2.bilateralFilter(img, 11, 17, 17)
        orig_img = img.copy()

        # Reset global variables
        drawing = False
        hsv_worked_flag = True
        points = []
        points_history = []
        # FIXME: Set trackbar to prediction
        cv2.setTrackbarPos(hl, wnd, 0)
        cv2.setTrackbarPos(hh, wnd, 179)
        cv2.setTrackbarPos(sl, wnd, 0)
        cv2.setTrackbarPos(sh, wnd, 255)
        cv2.setTrackbarPos(vl, wnd, 0)
        cv2.setTrackbarPos(vh, wnd, 255)

        # Close program if needed
        if exit_flag:
            break

        while (1):
            # convert from a BGR stream to an HSV stream
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # read trackbar positions for each trackbar
            hul = cv2.getTrackbarPos(hl, wnd)
            huh = cv2.getTrackbarPos(hh, wnd)
            sal = cv2.getTrackbarPos(sl, wnd)
            sah = cv2.getTrackbarPos(sh, wnd)
            val = cv2.getTrackbarPos(vl, wnd)
            vah = cv2.getTrackbarPos(vh, wnd)

            # make array for final values
            HSVLOW = np.array([hul, sal, val])
            HSVHIGH = np.array([huh, sah, vah])

            # create a mask for that range
            mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)

            res = cv2.bitwise_and(img, img, mask=mask)

            cv2.imshow(wnd, res)
            k = cv2.waitKey(10)
            if k == 13:     # Enter
                break
            if k == 8:      # Backspace
                hsv_worked_flag = False
                break
            if k == 27:     # Escape
                exit_flag = True
                break
            if drawing: # If user starts drawing points, then just exit the HSV mode
                hsv_worked_flag = False
                break

        if hsv_worked_flag:
            if not exit_flag:
                contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # Only biggest contour for now

                min_quad = None
                for cntr in contours:
                    if cv2.contourArea(cntr) > 1000:
                        min_quad = minEnclosingQuad(cntr, mask)

                # Create new points
                points_history.append(copy.deepcopy(points))

                # Add new 4 points!
                frame_idx = 0   # Since we just started
                points.append(box_to_list(min_quad)[0])
                draw_lines()

        while (1):  # While loop waiting for user to keep drawing more points / modifying them until they hit enter
            cv2.imshow('image', img)
            k = cv2.waitKey(10) & 0xFF
            if k == 13:     # Enter
                print(str(filename) + "\t" + str(len(points)) + "\t" + str(order_points(points)))
                outfile = open(outfile_name, "a")
                outfile.write(str(filename) + "\t" + str(len(points)) + "\t" + str(order_points(points)) + "\n")
                outfile.close()
                break
            if k == 27:     # Escape
                exit_flag = True
                break
            if k == ord('z'):
                print(points)
                # print(points_history)
                if len(points_history):
                    points = points_history.pop()
                else:
                    points = []
                # print(points)
                draw_lines()

cv2.destroyAllWindows()