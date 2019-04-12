import math
import os
import time

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import cv2
import numpy as np
import tensorflow as tf

# Import utilites
from ChessVisualiser.utils import label_map_util
from ChessVisualiser.utils import visualization_utils as vis_util

"""
def doCrop(_frame, rectangle):
def checkPointInRec(topLeft, bottomRight, point):
def getMiddles(im_width, im_height, boxes):
def getMiddle(im_width, im_height, box):
def doFilter(cropped, threshold, boxes, classes, scores):
def distance(p1, p2=(0, 0)):
def getCorners(im_width, im_height, boxes):
def drawCircle(img, point, radius=5, color=(255, 0, 0)):
def drawLine(img, line, color=(255, 0, 0)):
def getOrientation(img, corners):

"""


def crop(_frame):
    """
    Find the rectangle used for cropping

    Convert image the grayscale
    Convert image to black and white by applying a threshold
    Morph the image using an OPEN algorithm https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    Get all the contours
    Get the contour with the max area (The board)

    :param _frame:
    :return rectangle:
    """

    gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, 5)
    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(contour)


def doCrop(_frame, rectangle):
    """
    Crops an image using a rectangle

    :param _frame:
    :param rectangle:
    :return image:
    """
    height, width, _ = _frame.shape

    boundary = math.ceil((width * 0.01 + height * 0.01) / 2)

    left = rectangle[0] - boundary
    top = rectangle[1] - boundary
    right = rectangle[0] + rectangle[2] + boundary
    bottom = rectangle[1] + rectangle[3] + boundary

    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > width:
        right = width
    if bottom > height:
        bottom = height

    topLeft = (left, top)
    bottomRight = (right, bottom)

    # Crop around the rectangle
    cropped = _frame[topLeft[1]: bottomRight[1], topLeft[0]: bottomRight[0]].copy()
    return cropped


def checkPointInRec(topLeft, bottomRight, point):
    """ Checks if point is inside a box"""
    if topLeft[0] < point[0] < bottomRight[0] and topLeft[1] < point[1] < bottomRight[1]:
        return True
    return False


def getMiddles(width, height, boxes):
    """ Get all middle points of all boxes in the image"""
    midpoints = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        midpoint = (int((xmin * width + xmax * width) / 2), int((ymin * height + ymax * height) / 2))
        midpoints.append(midpoint)
    return midpoints


def getMiddle(width, height, box):
    """ Get middle point of a box in the image """
    ymin, xmin, ymax, xmax = box
    midpoint = (int((xmin * width + xmax * width) / 2), int((ymin * height + ymax * height) / 2))
    return midpoint


def doFilter(cropped, threshold, boxes, classes, scores):
    """
    Filter out all detections with a score less than the threshold
    And find multiple detections on the same piece, take the highest, remove the rest

    :param cropped:
    :param threshold:
    :param boxes:
    :param classes:
    :param scores:
    :return:
    """

    squeezed_boxes = np.squeeze(boxes)
    squeezed_scores = np.squeeze(scores)
    squeezed_classes = np.squeeze(classes).astype(np.int32)
    height, width, _ = cropped.shape

    # Find everything with a score of over the threshold
    for i in range(len(squeezed_scores)):
        if squeezed_scores[i] * 100 < threshold:
            keep = i
            break

    # Remove everything up to the thresholded detections
    squeezed_scores = squeezed_scores[:keep]
    squeezed_boxes = squeezed_boxes[:keep]
    squeezed_classes = squeezed_classes[:keep]

    # Find the center of every bounding box
    midpoints = getMiddles(width, height, squeezed_boxes)

    # Find all the pieces that have a center inside another bounding box
    # Check if its score is less that the other ones
    # If it is, mark it for removal
    to_remove = []
    for i in range(len(squeezed_boxes)):
        box = squeezed_boxes[i]
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
        for j in range(len(midpoints)):
            if i is not j:
                point = midpoints[j]
                if checkPointInRec((left, top), (right, bottom), point):
                    if squeezed_scores[j] > squeezed_scores[i]:
                        to_remove.append(i)
                        break

    # Remove all duplicates from the list
    to_remove = list(dict.fromkeys(to_remove))

    # Remove the duplicate weaker detections
    copy_boxes = []
    copy_scores = []
    copy_classes = []
    for i in range(len(squeezed_scores)):
        if i not in to_remove:
            copy_boxes.append(squeezed_boxes[i])
            copy_classes.append(squeezed_classes[i])
            copy_scores.append(squeezed_scores[i])

    return copy_boxes, copy_classes, copy_scores


def distance(p1, p2=(0, 0)):
    """
    Point Distance
    :param p1:
    :param p2:
    :return Float:
    """
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def getCorners(width, height, boxes):
    """
    Finds the points closest to each corner

    :param width:
    :param height:
    :param boxes:
    :return list of points:
    """
    topLeftP = (0, 0)
    topRightP = (width, 0)
    botRightP = (width, height)
    botLeftP = (0, height)

    # (Position in array, distance from respective points)
    topLeftClosest = (-1, height * 10)
    topRightClosest = (-1, height * 10)
    botRightClosest = (-1, height * 10)
    botLeftClosest = (-1, height * 10)

    for i in range(len(boxes)):
        box = boxes[i]
        midpoint = getMiddle(width, height, box)
        if distance(midpoint, topLeftP) < topLeftClosest[1]:
            topLeftClosest = (i, distance(midpoint, topLeftP))

        if distance(midpoint, topRightP) < topRightClosest[1]:
            topRightClosest = (i, distance(midpoint, topRightP))

        if distance(midpoint, botRightP) < botRightClosest[1]:
            botRightClosest = (i, distance(midpoint, botRightP))

        if distance(midpoint, botLeftP) < botLeftClosest[1]:
            botLeftClosest = (i, distance(midpoint, botLeftP))

    return [topLeftClosest[0], topRightClosest[0], botRightClosest[0], botLeftClosest[0]]


"""
VISUALISATION UTILITIESS
"""


def drawCircle(img, point, color=(255, 0, 0), radius=5):
    x = point[0]
    y = point[1]
    image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    np.copyto(img, np.array(image_pil))


def drawLine(img, line, color=(255, 0, 0), size=3):
    image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    draw.line(line, color, size)
    np.copyto(img, np.array(image_pil))


"""
END OF VISUALISATION UTILITIES
"""


def getAvgLuminance(img, boxes):
    """
    Take the middle of all detected objects, get the average luminance of all these pieces
    Luminance formula is
        Y = 0.2126 * Red + 0.7152 * Green + 0.0722 * Blue

    :param img:
    :param boxes:
    :return the average luminance of all pieces in the image:
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg_luminance = 0
    counter = 0
    height, width, _ = rgb.shape
    for mid in getMiddles(width, height, boxes):
        midY = int(mid[1])
        midX = int(mid[0])
        pixel = list(rgb[midY - 1:midY, midX - 1: midX][0][0])
        avg_luminance += 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
        counter += 1
    avg_luminance /= counter

    return avg_luminance


def getPieceColor(img, current, avg_luminance):
    """
    Computes the color of a piece by getting the luminance. A dark object has low luminance.
    If the pieces luminance is below the avg_luminance of all pieces in the image, it is black

    :param img:
    :param current:
    :param avg_luminance:
    :return piece color, 1 white, 0 black:
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    curY = int(current[1])
    curX = int(current[0])
    pixel = list(rgb[curY - 1:curY, curX - 1: curX][0][0])
    cur_luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
    return 1 if cur_luminance >= avg_luminance else 0


def getOrientation(img, corners, boxes):
    """
    Get the luminance / color of each piece and figure out the board orientation from it
    If there is a black piece in the top left, and black piece in top right, orientation is [0, 0, 1, 1]

    :param img:
    :param corners:
    :param boxes:
    :return list of 1's and 0's where 1 is white, 0 is black:
    """
    avg_luminance = getAvgLuminance(img, boxes)
    orientation = []
    for corner in corners:
        color = getPieceColor(img, corner, avg_luminance)
        orientation.append(color)

    return orientation


def beginVideoProcessing(PATH, user):
    VIDEO_PATH = os.path.join(PATH, 'input.mp4')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start = time.time()

    NET_PATH = os.path.dirname(os.path.realpath(__file__))
    NET_PATH = os.path.join(NET_PATH, 'static')
    NET_PATH = os.path.join(NET_PATH, 'net')

    MODEL_PATH = os.path.join(NET_PATH, 'frozen_inference_graph.pb')
    LABELS_PATH = os.path.join(NET_PATH, 'labelmap.pbtxt')

    NUM_CLASSES = 12

    label_map = label_map_util.load_labelmap(LABELS_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Get video
    cap = cv2.VideoCapture(VIDEO_PATH)
    # Get frame_spacing for video
    frame_spacing = cap.get(cv2.CAP_PROP_FPS) / 2

    rectangle = None
    while rectangle is None:
        # Read frame
        flag, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        # Get current frame position
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Check if frame is ready to be read
        if flag:
            # Get the rectangle cropping area
            rectangle = crop(frame)
        # Set frame back to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
        if rectangle is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_spacing / 4))

    frame = doCrop(frame, rectangle)
    height, width, _ = frame.shape
    video = cv2.VideoWriter(PATH + '/processed.mp4', 0x7634706d, 30, (width, height))

    while True:
        # Read in frame
        flag, frame = cap.read()
        # Get frame position
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Check if we're at a position to grab a frame
        # Check if frame is ready
        if flag:
            # frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            # Crop the frame to the rectangle
            cropped = doCrop(frame, rectangle)

            image_expanded = np.expand_dims(cropped, axis=0)
            # Perform the actual detection by running the model with the image as input

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            boxes, classes, scores = doFilter(cropped, 20, boxes, classes, scores)

            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                cropped,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.01,
                max_boxes_to_draw=50)

            for i in range(int(10)):
                video.write(cropped)

        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the current frame is final frame, stop
            break

        if flag:
            next_frame = pos_frame + frame_spacing
            if next_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT):
                next_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

    cap.release()
    video.release()
    end = time.time()
    print("It took {} seconds to process this video".format(end - start))
