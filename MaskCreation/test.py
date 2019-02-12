import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test.jpg')
height, width, channels = img.shape

colorCounts = []
colors = []
for i in range(0, height):
    for j in range(0, width):
        if str(img[i][j]) in colors:
            index = colors.index(str(img[i][j]))
            colorCounts[index] += 1
        else:
            colors.append(str(img[i][j]))
            colorCounts.append(1)

print ('next')
for a in range(len(colors)):
    print(str(colors[a]) + ' : ' + str(colorCounts[a]))

print(str(colors[0]) + ' : ' + str(colorCounts[0]))
print(str(colors[1]) + ' : ' + str(colorCounts[1]))
print(str(colors[2]) + ' : ' + str(colorCounts[2]))
print(str(colors[3]) + ' : ' + str(colorCounts[3]))