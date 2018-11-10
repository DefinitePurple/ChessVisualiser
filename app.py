import numpy as np
import random
import cv2  # pip install opencv-python
from matplotlib import pyplot as plt
import math as Math
from matplotlib import image as image
import easygui  # pip install easygui


def getCroppingRegion(img):
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    rectangle = cv2.boundingRect(contour)
    topLeft = (rectangle[0], rectangle[1])
    bottomRight = (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])
    # Crop around the rectangle
    return (topLeft[0], topLeft[1]), (bottomRight[0], bottomRight[1])


def getOrientation(img):
    img_width, img_width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T = np.mean(gray) + np.std(gray)
    T, binary = cv2.threshold(gray, thresh=T, maxval=255, type=cv2.THRESH_BINARY)

    firstSeg = binary[:img_width / 4, :img_width / 4].copy()
    secondSeg = binary[:img_width / 4, img_width / 2 + img_width / 4:].copy()
    thirdSeg = binary[img_width / 2 + img_width / 4:, img_width / 2 + img_width / 4:].copy()
    fourthSeg = binary[img_width / 2 + img_width / 4:, :img_width / 4].copy()

    firstSegMean = np.mean(firstSeg)
    secondSegMean = np.mean(secondSeg)
    thirdSegMean = np.mean(thirdSeg)
    fourthSegMean = np.mean(fourthSeg)
    segMeans = [firstSegMean, secondSegMean, thirdSegMean, fourthSegMean]
    segMeans.sort()

    whitest = segMeans[len(segMeans) - 1]
    secondWhitest = segMeans[len(segMeans) - 2]

    orientationArray = [0, 0, 0, 0]

    if whitest == firstSegMean:
        orientationArray[0] = 1
    elif whitest == secondSegMean:
        orientationArray[1] = 1
    elif whitest == thirdSegMean:
        orientationArray[2] = 1
    elif whitest == fourthSegMean:
        orientationArray[3] = 1

    if secondWhitest == firstSegMean:
        orientationArray[0] = 1
    elif secondWhitest == secondSegMean:
        orientationArray[1] = 1
    elif secondWhitest == thirdSegMean:
        orientationArray[2] = 1
    elif secondWhitest == fourthSegMean:
        orientationArray[3] = 1

    theta = 0
    if orientationArray == [1, 0, 0, 1]:
        theta = 90
    elif orientationArray == [1, 1, 0, 0]:
        theta = 180
    elif orientationArray == [0, 1, 1, 0]:
        theta = 270

    return theta


def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)

    S = gxx * gyy - gxy ** 2
    return S


# Opening an image from a file:
# source = cv2.imread("./Frames/0.jpg")
video = cv2.VideoCapture('./Video/1.mp4')
success, source = video.read()

#blur_img = cv2.blur(source, (3, 3))  # Blur it
saddle = getSaddle(source)
# pointOne, pointTwo = getCroppingRegion(source)
#
# cropped = source[pointOne[1]: pointTwo[1], pointOne[0]: pointTwo[0]].copy()
# orientation = getOrientation(cropped)
#
# while success:
#     success, frame = video.read()
#     pointOne, pointTwo = getCroppingRegion(frame)
#     if success:
#         cropped = frame[pointOne[1]: pointTwo[1], pointOne[0]: pointTwo[0]].copy()
#         height, width, _ = cropped.shape
#         rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), orientation, 1)
#         rotated = cv2.warpAffine(cropped, rotationMatrix, (height, width))
#
#         gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#         edges = cv2.Canny(blur, 0, 40, apertureSize=3)
#
#         combined = np.concatenate((edges, blur), axis=1)
#
#         cv2.imshow('combined', combined)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

plt.figure()
plt.subplot(1, 1, 1), plt.imshow(saddle, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()
cv2.destroyAllWindows()
