import cv2
import math
import time
import numpy as np
from matplotlib import pyplot as plt


def getContours(img):
    _, cnt, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt) <= 0:
        print('Could not find any contours')
        exit()
    return cnt


def crop(_frame):
    gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, 5)
    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours = getContours(binary)
    contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(contour)


def doCrop(_frame, rectangle):
    # Set the corners
    topLeft = (rectangle[0], rectangle[1])
    bottomRight = (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])
    # Crop around the rectangle
    cropped = _frame[topLeft[1]: bottomRight[1], topLeft[0]: bottomRight[0]].copy()
    return cropped


def beginVideoProcessing(videoPath):
    # Get video
    cap = cv2.VideoCapture(videoPath)
    # Get fps of video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get which frame to grab
    # ie: grab nth frame where n is framesToGrab
    # The divisor is the number of frames that will be grabbed in 1 second
    framesToGrab = math.floor(fps / 1)

    rectangle = None
    while rectangle is None:
        # Read frame
        flag, frame = cap.read()
        # Get current frame position
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Check if frame is ready to be read
        if flag:
            # Get the rectangle cropping area
            rectangle = crop(frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
        cv2.waitKey(1000)
    start = time.time()

    framesToKeepPos = []
    framesWithHands= []
    framesToDetectPos = []

    while True:
        # Read in frame
        flag, frame = cap.read()
        # Get frame position
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Check if we're at a position to grab a frame
        if pos_frame % framesToGrab == math.floor(framesToGrab / 2):
            # Check if frame is ready
            if flag:
                # Crop the frame to the rectangle
                cropped = doCrop(frame, rectangle)
                framesToKeepPos.append(pos_frame)
                cv2.imshow("frame", cropped)
                # plt.imshow(cropped)
                # plt.show()
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)
        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

    cv2.imwrite("./videos/test.jpg", cropped)

    end = time.time()
    print("It took {} seconds to process this video".format(end - start))
    # print("Frames to keep {}".format(framesToKeepPos))
    cap.release()


beginVideoProcessing("./videos/1.mp4")

# img = cv2.imread("./videos/1.jpg")
# crop(img)
