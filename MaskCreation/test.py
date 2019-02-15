import cv2
import numpy as np
from matplotlib import pyplot as plt


def getMask(lower, upper, img):
    lower = np.array(lower)
    upper = np.array(upper)
    return cv2.inRange(img, lower, upper)

print('Reading image')
source = cv2.imread('test.jpg')
height, width, channels = source.shape

source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)  # Convert source to RGB, down with BGR!
print('Doing mask stuff')
red_mask = getMask([190, 0, 0], [255, 100, 100], source)
blue_mask = getMask([0, 0, 190], [100, 100, 255], source)
green_mask = getMask([0, 190, 0], [100, 255, 100], source)
ROI = red_mask + blue_mask + green_mask
black_mask = getMask([0, 0, 0], [50, 50, 50], source)

output = source.copy()  # Copy the source image
output[np.where(red_mask != 0)] = (0, 0, 255)
output[np.where(green_mask != 0)] = (0, 255, 0)
output[np.where(blue_mask != 0)] = (255, 0, 0)
output[np.where(black_mask != 0)] = (0, 0, 0)

print('counting')
colorCounts = []
colors = []
for i in range(0, height):
    for j in range(0, width):
        if str(output[i][j]) in colors:
            index = colors.index(str(output[i][j]))
            colorCounts[index] += 1
        else:
            colors.append(str(output[i][j]))
            colorCounts.append(1)

print('next')
for a in range(len(colors)):
    print(str(colors[a]) + ' : ' + str(colorCounts[a]))

print(str(colors[0]) + ' : ' + str(colorCounts[0]))
print(str(colors[1]) + ' : ' + str(colorCounts[1]))
print(str(colors[2]) + ' : ' + str(colorCounts[2]))
print(str(colors[3]) + ' : ' + str(colorCounts[3]))
