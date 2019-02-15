#################################
#   Author: Daniel Fitzpatrick  #
#   sNo:    C15345046           #
#################################
# Step by Step
# 1) Convert image to HSV
# 2) Create red masks hsv ranges [0,50,50]->[10,100,100] and [170,100,100]->[180,255,255]
# 3) Create white masks hsv range [0,0,225]->[255, 40, 255]
# 4) Convert all pixels that are black in both masks to black in the output
# 5) Convert all pixels that are not black in the red mask to red
# 5) Convert all pixels that are not black in the white mask to blue (Easier for pixel comparison)
# 6) Get the laplacians of both masks and combine them together
# 7) Count colors of neighbours for a pixel then do stuff with those counts
# 8) Clear points that have close distance (1 will live)
# 9) Draw a circle around the points
###############################################
# In Depth:
# Ranges
#   In the hsv color space red wraps around 170 to 10 in the hue values so we need two masks and upper and lower one.
#   White wraps around 240 to 255 with a sat of 15 to make it easier to find however because of distortion,
#   we actually want to increase that boundary to allow for impurities
# Laplacian
#   The laplace operator to find the find boundarys within the masks. We apply it to the red and white mask so that
#   we can find where the boundaries of each color is which allows us to clean the image a bit and remove big blocks of
#   solid color. Small clumps of colors, such as stripes, should stay roughly the same
# Neighbours
#   For this I created my own function so it takes a bit of time to run (yay for loops!)
#   Essentially, all I'm doing is taking a pixel and counting how many blue, black, and red pixels are surrounding it
#   including itself. Then we do the following:
#   Check if the black count is less than 2 to allow some missing pixels
#   Check if white count and black count is greater than 0
#   Check if the white count and red count are both below the number of pixels / 1.75 rounded up
#   If these conditions are met, we add the pixels coords to a list
# Points
#   Go through the list of points and remove points with a distance close to each other keeping 1
###############################################
# Previous lookats
# Hough (pronounced Hawk) Line
#   I tried to use hough line to remove all vertical lines since wallys stripes are horizontal but I couldn't get 
#   this to work on every single line in the image, only some.
# My own line destroyers
#   I created my own functions (left in) to clear vertical lines and with the correct parameters when searching for neighbours
#   it actually worked out really well but it was incredibly slow due to searching every pixel so I decided to move on from this idea
################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from matplotlib import image as image
import easygui  # pip install easygui


def getMask(lower, upper, img):
    lower = np.array(lower)
    upper = np.array(upper)
    return cv2.inRange(img, lower, upper)


def clamp(val, max):
    if val < 0:
        return 0
    elif val > max:
        return max
    else:
        return val


def checkNeighbour(img, i, j, h, w, d):
    top = clamp(i - d, h)
    bot = clamp(i + d + 1, h)
    left = clamp(j - d, w)
    right = clamp(j + d + 1, w)
    new = img[top: bot, left:right]
    h, w, _ = new.shape

    red = list((new[:, :, 0] == 255))
    white = list((new[:, :, 2] == 255))

    redCount = np.count_nonzero(red)
    whiteCount = np.count_nonzero(white)
    combined = whiteCount + redCount
    blackCount = (h * w) - combined

    return redCount, whiteCount, blackCount


def clearVerticals(img, j, d):
    if d < 3:
        d = 3
    img = img[:, j:j + 1]
    reds = []
    whites = []
    for i in range(len(img)):
        pixel = list(img[i][0])
        if pixel[0] == 255:
            if len(whites) >= d:
                for c in range(len(whites)):
                    img[i - c - 1][0] = [0, 0, 0]
            whites = []
            reds.append(pixel)
        elif pixel[2] == 255:
            if len(reds) >= d:
                for c in range(len(reds)):
                    img[i - c - 1][0] = [0, 0, 0]
            reds = []
            whites.append(pixel)
        elif pixel == [0, 0, 0]:
            colors = []
            if len(reds) >= d:
                colors.extend(reds)
            elif len(whites) >= d:
                colors.extend(whites)

            if len(colors) >= d:
                for c in range(len(colors)):
                    img[i - c - 1][0] = [0, 0, 0]
            reds = []
            whites = []

    colors = []
    if len(reds) >= d:
        colors.extend(reds)
    elif len(whites) >= d:
        colors.extend(whites)

    if len(colors) >= d:
        for c in range(len(colors)):
            img[len(img) - c - 1][0] = [0, 0, 0]


# Find image
file = easygui.fileopenbox()
# If theres no file, exit the program
if file is None:
    print('Error: File not found')
    exit()
# Read image
source = cv2.imread(file)
if source is None:
    print('Error: Image not found')
    exit()

source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)  # Convert source to RGB, down with BGR!
source_hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)  # Convert to HSV

print('Creating masks')
lower_red_mask = getMask([0, 50, 50], [10, 100, 100], source_hsv)  # Red Hue is 170-180 and 0-10, bot masks are here
red_mask = getMask([170, 100, 100], [180, 255, 255], source_hsv) + lower_red_mask  # Get the red mask, hue: 170-180
white_mask = getMask([0, 0, 225], [255, 40, 255], source_hsv)  # White mask, hue: all, sat: some, value: 240-255

ROI = red_mask + white_mask  # Combine the red and white mask

output = source.copy()  # Copy the source image
output[np.where(ROI == 0)] = 0  # Set pixels to black where they are black in the mask

output[np.where(red_mask == 255)] = (255, 0, 0)  # Set pixels to red where they are white in the red mask
output[np.where(white_mask == 255)] = (0, 0, 255)  # Set pixels to blue where they are white in the white mask
height, width, _ = output.shape

# TODO Removing vertical above a certain length is actually incredibly accurate
# TODO with the right parameters when searching for possibly wallys
# out = output.copy()
#
# # Remove all vertical lines that are of either red or white(blue) and above a certain length
# print 'Clearing vertical lines'
# for a in range(width):
#     clearVerticals(out, a, 3)
#
#
# TODO Could not get HoughLineP to detect all lines in the image
# TODO Couln't figure out valid parameters to get all vertical lines
# lines = cv2.HoughLinesP(red_mask, 1, np.pi / 180, 10, 4, 0)
# for line in lines:
#     for x1, y1, x2, y2 in line:
#         angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
#         if 90 or 270:
#             cv2.line(out, (x1, y1), (x2, y2), (0, 0, 0), 1)
#
# lines = cv2.HoughLinesP(white_mask, 1, np.pi / 180, 10, 4, 0)
# for line in lines:
#     for x1, y1, x2, y2 in line:
#         angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
#         if 90 or 270:
#             cv2.line(out, (x1, y1), (x2, y2), (0, 0, 0), 1)

# Use laplacian on the red and white masks to highlight vertical and horizontal lines
laplacianW = cv2.Laplacian(white_mask, cv2.CV_64F)
laplacianW[np.where(laplacianW != 0)] = 1
laplacianR = cv2.Laplacian(red_mask, cv2.CV_64F)
laplacianR[np.where(laplacianR != 0)] = 1
laplacian = laplacianW + laplacianR
output[np.where(laplacian != 2)] = 0

# Find clusters of only red and white pixels with an allowance of 1 black pixel
# This is the slowest point of the process because its going through the image pixel by pixel
print 'Getting possible Wallys'
points = []
for a in (range(height)):
    for b in range(width):
        redCount, whiteCount, blackCount = checkNeighbour(output, a, b, height, width, 1)
        if blackCount < 2 \
                and 0 < whiteCount <= math.ceil((redCount + whiteCount + blackCount) / 1.75) \
                and 0 < redCount <= math.ceil((redCount + whiteCount + blackCount) / 1.75):
            points.append([a, b])


# Use point distances to remove similar points
print 'Clearing similar points'
points = sorted(points, key=lambda point: point[0])
check = []
check.extend(points)
for p1 in points:
    for a in range(0, len(points) - 1):
        p2 = points[a]
        if p1 != p2 and p2 in check and p1 in check:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if dist < 30.0:
                check.remove(p2)

# Draw circles around all the possible wallys
print 'Found {} possible wallys'.format(len(check))
h, w, _ = source.shape
for p in check:
    cv2.circle(source, (p[1], p[0]), h/25, (0, 255, 0), 2)
    cv2.circle(source, (p[1], p[0]), h/25 + 2, (0, 0, 0), 2)

plt.imshow(source)
plt.show()
