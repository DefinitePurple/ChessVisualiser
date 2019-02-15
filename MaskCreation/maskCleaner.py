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


# Read image
source = cv2.imread('test.jpg')

source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)  # Convert source to RGB, down with BGR!

red_mask = getMask([210, 0, 0], [255, 40, 40], source)
blue_mask = getMask([0, 0, 210], [40, 40, 255], source)
green_mask = getMask([0, 210, 0], [40, 255, 40], source)
ROI = red_mask + blue_mask + green_mask

black_mask = getMask([0, 0, 0], [30, 30, 30], source)

output = source.copy()  # Copy the source image
output[np.where(red_mask != 0)] = (0, 0, 255)
output[np.where(green_mask != 0)] = (0, 255, 0)
output[np.where(blue_mask != 0)] = (255, 0, 0)
output[np.where(black_mask != 0)] = (0, 0, 0)

cv2.imwrite('test_cleaned.jpg', output)

plt.imshow(black_mask)
plt.show()

