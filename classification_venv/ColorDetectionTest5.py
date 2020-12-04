import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from os import listdir
from os.path import isfile, join
import os

np.seterr(all='raise')  # So divide by zero comes out as exception


def displayBGR(image):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def displayGRAY(image):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    plt.show()


# This function will create a line that starts at (x1, y1) and pass through (x2,y2), continuing to
# the boundaries of the image, which are defined by 0 and maxX and maxY.  Returns an array of points
# (pixels) on that line.
def get_line_long(x1, y1, x2, y2, maxX, maxY):
    minX = 0
    minY = 0

    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1 = -x1    # Invert x and y so line is always going in positive direction
        x2 = -x2
        y1 = -y1
        y2 = -y2
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1

    complete = False
    x = x1  # Start at x1
    while not complete:
        # if not rev:
        if issteep:
            if not rev:
                points.append((y, x))
                if (x <= minY) or (x >= maxY) or (y <= minX) or (y >= maxX):
                    complete = True
            elif rev:
                points.append((-y, -x))
                if (x >= -minY) or (x <= -maxY) or (y >= -minX) or (y <= -maxX):
                    complete = True
        else:
            if not rev:
                points.append((x, y))
                if (x <= minX) or (x >= maxX) or (y <= minY) or (y >= maxY):
                    complete = True
            elif rev:
                points.append((-x, -y))
                if (x >= -minX) or (x <= -maxX) or (y >= -minY) or (y <= -maxY):
                    complete = True
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
        x += 1  # Increment x

    return points


def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


# (X[i], Y[i]) are coordinates of i'th point.
def polygonArea(X, Y):
    n = len(X)
    # Initialize area
    area = 0.0

    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0, n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i  # j is previous vertex to i

    # Return absolute value
    return abs(area / 2.0)


# Just calls polygonArea
def area_of_box(box):
    return polygonArea((box[0][0], box[1][0], box[2][0], box[3][0]), (box[0][1], box[1][1], box[2][1], box[3][1]))


# Like area_of_box but so it can easily be called to check a new line
# Side is the side you're working on, point1 and point2 are the new points you're testing,
# box is the original box that you're replacing one edge of
def area_of_new_box(side, point1, point2, box):
    # IMPORTANT: point2 must follow point1 clockwise
    if side == 'left':
        return polygonArea((point1[0], point2[0], box[2][0], box[3][0]), (point1[1], point2[1], box[2][1], box[3][1]))
    elif side == 'top':
        return polygonArea((box[0][0], point1[0], point2[0], box[3][0]), (box[0][1], point1[1], point2[1], box[3][1]))
    elif side == 'right':
        return polygonArea((box[0][0], box[1][0], point1[0], point2[0]), (box[0][1], box[1][1], point1[1], point2[1]))
    elif side == 'bottom':
        return polygonArea((point2[0], box[1][0], box[2][0], point1[0]), (point2[1], box[1][1], box[2][1], point1[1]))
    else:
        exit('Error: bad side in area_of_new_box: ' + str(side))


# Image passed in must be GRAY, not BGR
# Returns false if no conflicts
def check_box_for_conflicts(box, img):
    left_side = get_line(box[0][0], box[0][1], box[1][0], box[1][1])
    top_side = get_line(box[1][0], box[1][1], box[2][0], box[2][1])
    right_side = get_line(box[2][0], box[2][1], box[3][0], box[3][1])
    bottom_side = get_line(box[3][0], box[3][1], box[0][0], box[0][1])
    all_sides = left_side + top_side + right_side + bottom_side
    for pix in all_sides:
        if img[pix[1]][pix[0]] != 0:
            # print("Problem: " + str(pix[1]) + "," + str(pix[0]]))
            return True
    return False


# Like check_box_for_conflicts, but checks one line (pass in two points)
# Image passed in must be GRAY, not BGR
# Returns false if no conflicts
def check_line_for_conflicts(point1, point2, img):
    line = get_line(point1[0], point1[1], point2[0], point2[1])
    for pix in line:
        if img[pix[1]][pix[0]] != 0:
            # print("Problem: " + str(pix[1]) + "," + str(pix[0]]))
            return True
    return False


# First pushes the edge out until it doesn't conflict, then rotates right and then left to find minimum area
def rotateCaliperLine(side, top_line, bottom_line, starting_top_point, starting_bottom_point, box, img):
    min_area = 999999999  # This will contain the minimum area of the box
    min_points = None  # This will be an array of points that give the min_area

    # Display the box before rotation (optional)
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_bgr, [box], 0, (0, 0, 255), 1)
    # displayBGR(img_bgr)

    # Check starting point
    complete = False
    while not complete:
        # If there are conflicts
        try:
            if check_line_for_conflicts(top_line[starting_top_point], bottom_line[starting_bottom_point], img):
                # If so, push back and repeat
                starting_top_point += 1
                starting_bottom_point += 1
                # If can't push back further, you'll error out in the except block
            # Once you're not conflicting get your starting min_area
            else:
                point1 = top_line[starting_top_point]
                point2 = bottom_line[starting_bottom_point]
                min_area = area_of_new_box(side, point1, point2, box)
                min_points = [point1, point2]
                complete = True
        except IndexError:
            exit("ERROR: for side " + str(
                side) + " we couldn't find a starting line that didn't conflict and was in bounds")
    # print('* Starting point: ' + str(top_line[starting_top_point]) + " " + str(bottom_line[starting_bottom_point]) + " *")

    # Display output lines! (Optional)
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.line(img_bgr, min_points[0], min_points[1], (0, 255, 0), 1)
    # displayBGR(img_bgr)

    # Then rotate right (by decrementing the bottom)
    #    /------/ | ->
    #   /------/  |
    #  /------/   |
    # /______/    | <-
    top_point = starting_top_point
    bottom_point = starting_bottom_point
    complete = False
    while not complete:
        try:
            bottom_point -= 1  # Decrement the bottom point
            # Shift right until you don't conflict
            while check_line_for_conflicts(top_line[top_point], bottom_line[bottom_point], img):
                top_point += 1
                bottom_point += 1
            # Now that you don't conflict, see if it's a new minimum area!
            point1 = top_line[top_point]
            point2 = bottom_line[bottom_point]
            if area_of_new_box(side, point1, point2, box) <= min_area:  # FIXME: <= or < ?
                min_area = area_of_new_box(side, point1, point2, box)
                min_points = [point1, point2]
                # print("New min points: " + str(min_points))
        except IndexError:
            # Go until your top_point goes out of bounds
            complete = True

    # Go back to start and rotate left
    top_point = starting_top_point
    bottom_point = starting_bottom_point
    complete = False
    while not complete:
        try:
            top_point -= 1  # Decrement the *TOP* point
            # Shift right until you don't conflict
            while check_line_for_conflicts(top_line[top_point], bottom_line[bottom_point], img):
                top_point += 1
                bottom_point += 1
            # Now that you don't conflict, see if it's a new minimum area!
            point1 = top_line[top_point]
            point2 = bottom_line[bottom_point]
            if area_of_new_box(side, point1, point2, box) <= min_area:  # FIXME: <= or < ?
                min_area = area_of_new_box(side, point1, point2, box)
                min_points = [point1, point2]
                # print("New min points: " + str(min_points))
        except IndexError:
            # Go until your top_point goes out of bounds
            complete = True

    # Display output lines! (Optional)
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.line(img_bgr, min_points[0], min_points[1], (0, 255, 0), 1)
    # displayBGR(img_bgr)

    return min_points


# Takes in which side of box to work with.  It rotates that side as far as it can to both degrees,
# optimizing for minimum box area
def rotateCaliper(side, box, img):
    top_line = None  # Will be an array of points
    bottom_line = None  # Will be an array of points
    relative_top_idx = None
    relative_bottom_idx = None
    rel_top_far_idx = None  # The index of far top corner
    rel_bot_far_idx = None  # Index of far bottom corner

    if side == 'right':

        relative_top_idx = 2
        relative_bottom_idx = 3
        rel_top_far_idx = 1
        rel_bot_far_idx = 0
    elif side == 'bottom':
        relative_top_idx = 3
        relative_bottom_idx = 0
        rel_top_far_idx = 2
        rel_bot_far_idx = 1
    elif side == 'left':
        relative_top_idx = 0
        relative_bottom_idx = 1
        rel_top_far_idx = 3
        rel_bot_far_idx = 2
    elif side == 'top':
        relative_top_idx = 1
        relative_bottom_idx = 2
        rel_top_far_idx = 0
        rel_bot_far_idx = 3
    else:
        exit("In function rotateCaliper, the input side is not recognized: " + str(side))

    # Use get_line_long to get top and bottom line:
    top_line = get_line_long(box[rel_top_far_idx][0], box[rel_top_far_idx][1], box[relative_top_idx][0], box[relative_top_idx][1], img.shape[1]-1, img.shape[0]-1)
    bottom_line = get_line_long(box[rel_bot_far_idx][0], box[rel_bot_far_idx][1], box[relative_bottom_idx][0], box[relative_bottom_idx][1], img.shape[1]-1, img.shape[0]-1)

    # FIXME: Remove this try block later once you know it's always true
    try:  # Ensure that the arrays are in the right direction
        top_closePoint = top_line.index((box[relative_top_idx][0], box[relative_top_idx][1]))
        top_farPoint = top_line.index((box[rel_top_far_idx][0], box[rel_top_far_idx][1]))
        bot_closePoint = bottom_line.index((box[relative_bottom_idx][0], box[relative_bottom_idx][1]))
        bot_farPoint = bottom_line.index((box[rel_bot_far_idx][0], box[rel_bot_far_idx][1]))
        if top_closePoint < top_farPoint:
            top_line.reverse()  # Reverse the array so that incrementing the index will go away from our close point
            print("REVERSING TOP ARRAY")
        if bot_closePoint < bot_farPoint:
            bottom_line.reverse()  # Do the same for bottom array as well
            print("REVERSING BOTTOM ARRAY")
    except ValueError:
        # This always fails when one of the points is outside - of the box, FIXME later?
        # exit("ERROR: Point not found on either top or bottom line...")
        print("ERROR: Point not found within bounds")
        return

    starting_top_point = top_line.index((box[relative_top_idx][0], box[relative_top_idx][1]))
    starting_bottom_point = bottom_line.index((box[relative_bottom_idx][0], box[relative_bottom_idx][1]))

    min_points = rotateCaliperLine(side, top_line, bottom_line, starting_top_point, starting_bottom_point, box, img)

    # Update box with new minimum area points
    box[relative_top_idx][0] = min_points[0][0]
    box[relative_top_idx][1] = min_points[0][1]
    box[relative_bottom_idx][0] = min_points[1][0]
    box[relative_bottom_idx][1] = min_points[1][1]


# Will minimize box around a non-black object in img
# Works by attempting to rotate each edge for minimal area
def geometricMinimizeQuad(box, img):
    loops = 3
    min_box = np.copy(box)
    for i in range(loops):
        rotateCaliper('right', min_box, img)
        rotateCaliper('bottom', min_box, img)
        rotateCaliper('left', min_box, img)
        rotateCaliper('top', min_box, img)
    return min_box


# img must be GRAY, not BGR
# This implements an algorithm to shrink the bounding box around a non-black object in img
# Currently calls geometricMinimizeQuad, originally it called recursive_min_box but it wasn't
# very efficient so that method is not recommended.
def minEnclosingQuad(cnt, img):
    # Fill in contour of picture with white
    cv2.fillPoly(img, pts=[cnt], color=(255, 255, 255))

    # Get coordinates
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Display the box (optional)
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_bgr, [box], 0, (0, 0, 255), 1)
    # displayBGR(img_bgr)

    minBox = geometricMinimizeQuad(box, img)
    # print("done!")

    # Display the box (optional)
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_bgr, [minBox], 0, (0, 0, 255), 1)
    # displayBGR(img_bgr)

    return minBox


# comp_file = "Python_Matching/raw/fronts/bandw2all.png"
# comp_image = cv2.imread(comp_file)
# comp_image = cv2.cvtColor(comp_image, cv2.COLOR_BGR2GRAY)
# _, comp_binary = cv2.threshold(comp_image, 225, 255, cv2.THRESH_BINARY)
#
# my_path = 'Python_Matching/raw/fronts/'
# onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]
# os.chdir(my_path)
#
# onlyfiles.remove('bandw1all.jpg')
# onlyfiles.remove('bandw1.jpg')
# onlyfiles.remove('bandw2all.png')
# onlyfiles.remove('output.png')
# onlyfiles.remove('yellowTest.jpg')
#
# # for filename in onlyfiles:
# for i in range(0, 1):
#     filename = "IMG_1804.jpg"
#     print(filename)
#     image = cv2.imread(filename)
#     image = imutils.resize(image, height=240)
#     image = cv2.bilateralFilter(image, 11, 17, 17)
#     # displayBGR(image)
#
#     # create NumPy arrays from the boundaries
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     hsv_lower = np.array([15, 0, 90], dtype="uint8")
#     hsv_upper = np.array([40, 255, 255], dtype="uint8")
#     # find the colors within the specified boundaries and apply the mask
#     mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
#     # mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     output = cv2.bitwise_and(image, image, mask=mask)
#     hsv_output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
#     # show the images
#     # displayGRAY(mask)
#     # displayBGR(output)
#
#     contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(contours)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # Only biggest contour for now
#
#     min_quad = None
#     for cntr in contours:
#         if cv2.contourArea(cntr) > 1000:
#             min_quad = minEnclosingQuad(cntr, mask)
#
#     cv2.drawContours(image, [min_quad], 0, (0, 0, 255), 2)
#     displayBGR(image)
#
#     cv2.imwrite("sample_output.png", image)




    # lower = np.array([20, 60, 90], dtype="uint8")
    # upper = np.array([255, 255, 255], dtype="uint8")
    # lower = hsv_lower
    # upper = hsv_upper
    # thresh = np.array([0, 0, 0], dtype="uint8")
    # thresh_lo = np.array([0, 0, 0], dtype="uint8")
    # thresh_hi = np.array([0, 0, 0], dtype="uint8")
    #
    # array_of_tuples = [[], [], []]
    #
    # # Plot histogram then use Otsu's method to split into multi-level threshold
    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([hsv_output], [i], None, [256], [1, 256])
    #
    #     thresh[i] = otsu_thresh(histr, lower[i], upper[i])
    #     thresh_lo[i] = otsu_thresh(histr, lower[i], thresh[i])
    #     thresh_hi[i] = otsu_thresh(histr, thresh[i], upper[i])
    #
    #     array_of_tuples[i].append((lower[i], upper[i]))
    #     array_of_tuples[i].append((lower[i], thresh[i]))
    #     array_of_tuples[i].append((thresh[i], upper[i]))
    #
    #     # array_of_tuples[i].append((thresh_lo[i], thresh_hi[i]))
    #     # array_of_tuples[i].append((lower[i], thresh_hi[i]))
    #     # array_of_tuples[i].append((thresh_lo[i], upper[i]))
    #     # array_of_tuples[i].append((lower[i], thresh_lo[i]))
    #     # array_of_tuples[i].append((thresh_lo[i], thresh[i]))
    #     # array_of_tuples[i].append((thresh[i], thresh_hi[i]))
    #     # array_of_tuples[i].append((thresh_hi[i], upper[i]))
    #
    #     plt.plot(histr, color=col)
    #     plt.axvline(x=thresh[i])
    #     plt.axvline(x=thresh_lo[i])
    #     plt.axvline(x=thresh_hi[i])
    #     plt.xlim([0, 256])
    #     plt.show()
    #
    # best_contour = None
    # best_contours = None
    # lowest_val = 2.41
    # best_idx = 0
    #
    # for (r_lo, r_hi) in array_of_tuples[2]:
    #     for (g_lo, g_hi) in array_of_tuples[1]:
    #         for (b_lo, b_hi) in array_of_tuples[0]:
    #             low = np.array([b_lo, g_lo, r_lo], dtype="uint8")
    #             upp = np.array([b_hi, g_hi, r_hi], dtype="uint8")
    #             mask = cv2.inRange(output, low, upp)
    #             output = cv2.bitwise_and(output, image, mask=mask)
    #             displayBGR(output)
    #
    #             contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #             contours = imutils.grab_contours(contours)
    #             contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    #
    #             counter = 0
    #             for c in contours:
    #                 if cv2.contourArea(c) > 1000:
    #                     # Match shapes using the huMoments of the contour / binary
    #                     d2 = cv2.matchShapes(c, comp_binary, cv2.CONTOURS_MATCH_I2, 0)
    #                     if d2 < 2.45:
    #                         match = ' it\'s a match!'
    #                         print(str(b_lo) + "-" + str(b_hi) + " " + str(g_lo) + "-" + str(g_hi) + " " + str(
    #                             r_lo) + "-" +
    #                               str(r_hi) + " " + str(d2) + match + " " + str(cv2.contourArea(c)))
    #                         displayBGR(cv2.drawContours(output.copy(), contours, counter, (0, 0, 255), 2))
    #                     # if d2 < lowest_val:
    #                     #     lowest_val = d2
    #                     #     best_contour = c
    #                     #     best_contours = contours
    #                     #     best_idx = counter
    #                 counter += 1
    #
    # # if best_contour is not None:
    # #     print(str(lowest_val) + "    " + str(cv2.contourArea(best_contour)))
    # #     displayBGR(cv2.drawContours(image.copy(), best_contours, best_idx, (0, 0, 255), 2))
    #
    # # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # # edged = cv2.Canny(gray, 30, 200)
    # # displayGRAY(edged)
    # #
    # # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # # edged = cv2.Canny(gray, 30, 200)
    # # displayGRAY(edged)
    #
    # # contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # contours = imutils.grab_contours(contours)
    # # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    # # displayBGR(cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 2))
    # #
    # # for c in contours:
    # #     rect = cv2.minAreaRect(c)
    # #     box = cv2.boxPoints(rect)
    # #     box = np.int0(box)
    # #     displayBGR(cv2.drawContours(image, [box], 0, (0, 0, 255), 4))
    # #
    # #     # Match shapes using the huMoments of the contour / binary
    # #     d2 = cv2.matchShapes(c, comp_binary, cv2.CONTOURS_MATCH_I2, 0)
    # #     match = ''
    # #     if d2 < 2.41:
    # #         match = ' it\'s a match!'
    # #     print(str(d2) + match)
