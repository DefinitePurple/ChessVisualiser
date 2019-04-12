import math
import os
import time

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as Font
import cv2
import numpy as np
import tensorflow as tf
# Import utilites
from utils import label_map_util
from utils import visualization_utils


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
def drawText(img, point, message, fill=(0, 0, 0), size=40):
    image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
    d = ImageDraw.Draw(image_pil)
    font = Font.truetype(font="arial.ttf", size=size)
    d.text(point, message, font=font, fill=fill)
    np.copyto(img, np.array(image_pil))


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

def perpDistFromLine(line, point):
    """
    Finds the perpindicular distance of a point from a line

    :param line:
    :param point:
    :return Float:
    """
    b = point
    a = line[0]
    c = line[1]

    ab = distance(a, b)  # left line point to point
    ca = distance(a, c)  # the line distance
    cb = distance(c, b)  # right line point to point

    top = (cb ** 2) + (ca ** 2) - (ab ** 2)
    bot = 2 * ca * cb

    alpha = math.acos(top / bot)
    yeeta = math.radians(90)
    right = cb / math.sin(yeeta)
    left = math.sin(alpha)

    return left * right


def bubbleSort(list, midpoints, reverse=False, axis=0):
    """
    Sort points in a list
    axis states if they are to be sorted by their x or y axis

    :param list:
    :param midpoints:
    :param reverse:
    :param axis:
    :return:
    """
    for passnum in range(len(list) - 1, 0, -1):
        for i in range(passnum):
            p1 = midpoints[list[i]][axis]
            p2 = midpoints[list[i + 1]][axis]
            if p1 > p2 and not reverse:
                temp = list[i]
                list[i] = list[i + 1]
                list[i + 1] = temp
            elif p1 < p2 and reverse:
                temp = list[i + 1]
                list[i + 1] = list[i]
                list[i] = temp
    return list


def bubbleSortLines(list, reverse=False):
    """
    Sort lines by their middle points

    :param list:
    :param reverse:
    :return Sorted List:
    """
    for passnum in range(len(list) - 1, 0, -1):
        for i in range(passnum):
            line1 = list[i]
            line2 = list[i + 1]
            mid1 = lineMid(line1)
            mid2 = lineMid(line2)

            if mid1 > mid2 and not reverse:
                temp = list[i]
                list[i] = list[i + 1]
                list[i + 1] = temp
            elif mid1 < mid2 and reverse:
                temp = list[i + 1]
                list[i + 1] = list[i]
                list[i] = temp
    return list


def lineMid(line):
    """
    Calculate the middle point of a line

    :param line:
    :return Point:
    """
    mid = ((line[0][0] + line[1][1]) / 2, (line[1][0] + line[1][0]) / 2)
    return mid


def line_intersection(line1, line2):
    """
    Calculate the intersection point of two lines

    :param line1:
    :param line2:
    :return Point:
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def horizontalOrVertical(orientation):
    """
    Figure out if the board is vertical or horizontal.

    :param orientation:
    :return Integer:
    """
    if orientation == [1, 0, 0, 1] or orientation == [0, 1, 1, 0]:
        return 0  # Vertical
    elif orientation == [0, 0, 1, 1] or orientation == [1, 1, 0, 0]:
        return 1  # Horizontal
    else:
        return -1  # Error


def whereIsA1(orientation):
    """
    Returns position in corners_mids array of which corner is 'a1'
    Starts clockwise from top left

    :param orientation:
    :return Integer notating where a1 is:
    """
    if orientation == [1, 0, 0, 1]:
        return 0  # Top Left
    elif orientation == [1, 1, 0, 0]:
        return 1  # Top Right
    elif orientation == [0, 1, 1, 0]:
        return 2  # Bottom Right
    elif orientation == [0, 0, 1, 1]:
        return 3  # Bottom Left
    else:
        return -1


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


def sortFiles(files, reverse=False):
    """
    ONLY USED WHEN READING IN THE IMAGE SEQUENES
    Sort a list of files by their name
    :param files:
    :param reverse:
    :return:
    """
    for passnum in range(len(files) - 1, 0, -1):
        for i in range(passnum):
            p1 = files[i].split('.')[0]
            p2 = files[i + 1].split('.')[0]
            if int(p1) > int(p2) and not reverse:
                temp = files[i]
                files[i] = files[i + 1]
                files[i + 1] = temp

            elif int(p1) < int(p2) and reverse:
                temp = files[i + 1]
                files[i + 1] = files[i]
                files[i] = temp

    return files


def convertToBoardMapping(diction):
    """
    Maps the results of compareAgainstBoard to a multidimensionalArray

    :param diction:
    :return a multidimensional array where ? notates pieces and - notates empty squares:
    """

    temp = []
    for i in range(8):
        row = []
        for j in range(8):
            row.append('-')
        temp.append(row)

    for p in diction:
        temp[p[0]][p[1]] = '?'
    return temp


def compareAgainstBoard(middles, board):
    """
    Finds the intersection point (square) each piece mid point is closest to

    :param middles:
    :param board:
    :return list of array coordinates where the coordinates map back to a position in the board array:
    """

    diction = []
    for i in range(len(middles)):
        mid = middles[i]
        closest = 1000000000
        square = [0, 0]
        for row in range(len(board)):
            for col in range(len(board[row])):
                point = board[row][col]
                dist = distance(mid, point)
                if dist < closest:
                    closest = dist
                    square = [row, col]
        diction.append(square)
    return diction


def compareAgainstPreviousState(current, previous):
    """
    Receives the current board state acquired from convertToBoardMapping and the previous board state
    This will keep track of piece movement. Converts '?' into actual pieces
    The first 'previous' state will be the starting position which already tells us what pieces are

    :param current:
    :param previous:
    :return board state with pieces:
    """
    move = (0, 0)
    for i in range(len(current)):
        for j in range(len(current[i])):
            cur = current[j][i]
            prev = previous[j][i]

            if prev is not '-' and cur is not '-':
                current[j][i] = prev
            elif prev is not '-' and cur is '-':
                move = (j, i)

    found = False
    i = 0
    while not found:
        for j in range(len(current[i])):
            cur = current[j][i]
            prev = previous[move[0]][move[1]]
            if cur is '?':
                current[j][i] = prev
                found = True
                break
        i += 1
    return current


def convertToFEN(state):
    """
    To convert the board state to a string, we just need to iterate through each 'rank' in the array
    and append it to a string

    :param state:
    :return FEN:
    """
    FEN = ""
    for i in range(len(state) - 1, -1, -1):
        counter = 0
        for j in range(len(state[i])):
            if state[j][i] is not '-':
                if counter is not 0:
                    FEN += str(counter)
                    counter = 0
                FEN += state[j][i]
            else:
                counter += 1
        if counter is not 0:
            FEN += str(counter)
        if i is not 0:
            FEN += '/'
    return FEN


def beginVideoProcessing(VIDEO_PATH, OUT_PATH, rotato):
    """
    INITLISATION
    Initialise all tensorflow paths and attributes
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    start = time.time()
    NET_PATH = os.path.dirname(os.path.realpath(__file__))
    NET_PATH = os.path.join(NET_PATH, 'static')
    NET_PATH = os.path.join(NET_PATH, 'net')

    MODEL_PATH = os.path.join(NET_PATH, 'frozen_inference_graph.pb')
    LABELS_PATH = os.path.join(NET_PATH, 'labelmap.pbtxt')

    NUM_CLASSES = 6

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
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    files = sorted(os.listdir(VIDEO_PATH))
    path = VIDEO_PATH + files[0]
    files.pop(0)
    img = cv2.imread(path)
    for i in range(rotato):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rectangle = crop(img)
    cropped = doCrop(img, rectangle)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'cropped'), cropped)

    image_expanded = np.expand_dims(cropped, axis=0)
    height, width, _ = cropped.shape

    """
    DETECTION
    Perform the actual detection
    boxes - bounding boxes of the detections
    scores - scores relating to the bounding boxes found
    classes - classification of the found box ie: chess_knight
    num - number of detections found
    """
    inputs = [detection_boxes, detection_scores, detection_classes, num_detections]
    (boxes, scores, classes, num) = sess.run(
        inputs,
        feed_dict={image_tensor: image_expanded})

    labels_img = cropped.copy()
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        labels_img,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.01,
        max_boxes_to_draw=50)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'labels'), labels_img)

    """
    FILTER
    Filter each bounding box to take the highest scoring of each overlapping box
    If the center point of a box is inside another box but its score is smaller, remove it  
    """
    boxes, classes, scores = doFilter(cropped, 20, boxes, classes, scores)
    labels_filtered_img = cropped.copy()
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        labels_filtered_img,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.01,
        max_boxes_to_draw=50)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'labels_filtered'), labels_filtered_img)

    """
    MIDDLES - NOT NECESSARY
    Get ALL the middle points from the boxes
    Draw them onto the image 
    """
    middles = getMiddles(width, height, boxes)
    middles_img = cropped.copy()
    for mid in middles:
        drawCircle(middles_img, mid, (0, 255, 0), 10)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'middles'), middles_img)

    """
    CORNERS
    Get the array index of each corner point
    Get the center point of each corner
    Append the center point to a list to hold the corner mid points    
    """
    corners = getCorners(width, height, boxes)
    corner_img = cropped.copy()
    corners_mids = []
    for i in corners:
        tmp = getMiddle(width, height, boxes[i])
        corners_mids.append(tmp)
        drawCircle(corner_img, tmp, (0, 255, 0), 10)

    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'corners'), corner_img)

    """
    ORIENTATION
    Convert image to RGB
    Get the center point of each corner piece
    Check if the pixel at the center is closer to white or black
    White is 1
    Black is 0
    """

    orientation = getOrientation(cropped, corners_mids, boxes)
    a1 = whereIsA1(orientation)
    """
    EDGES
    Create lines between the corner points
    Check the perpendicular distance of each box mid point from each line
    If its closer to the top line/edge, add it to that lines array
    """
    # Arrays meant to hold the edge points
    top_points = []
    right_points = []
    bot_points = []
    left_points = []

    # Append the corners to the arrays set to hold the edges
    top_points.append(corners[0])
    top_points.append(corners[1])
    right_points.append(corners[1])
    right_points.append(corners[2])
    bot_points.append(corners[2])
    bot_points.append(corners[3])
    left_points.append(corners[3])
    left_points.append(corners[0])

    # Get all the midpoints
    midpoints = getMiddles(width, height, boxes)
    # This is the distance to compare against when checking the distance from a line
    check = distance((0, 0), (width / 20, height / 20))
    for i in range(len(boxes)):
        if i not in corners:
            distTop = perpDistFromLine((corners_mids[0], corners_mids[1]), midpoints[i])
            distBot = perpDistFromLine((corners_mids[2], corners_mids[3]), midpoints[i])
            distLeft = perpDistFromLine((corners_mids[0], corners_mids[3]), midpoints[i])
            distRight = perpDistFromLine((corners_mids[1], corners_mids[2]), midpoints[i])
            if distTop < check:
                top_points.append(i)
            if distBot < check:
                bot_points.append(i)
            if distLeft < check:
                left_points.append(i)
            if distRight < check:
                right_points.append(i)

    # Draw a circle on each edge point
    edge_img = cropped.copy()
    for i in left_points:
        drawCircle(edge_img, midpoints[i], (0, 255, 0), 10)
    for i in right_points:
        drawCircle(edge_img, midpoints[i], (0, 255, 0), 10)
    for i in top_points:
        drawCircle(edge_img, midpoints[i], (0, 255, 0), 10)
    for i in bot_points:
        drawCircle(edge_img, midpoints[i], (0, 255, 0), 10)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'edges'), edge_img)

    """
    LINES
    In order to figure out the approximate center point of each board square, 
    it sorts each point so that we can connect them and create as straight a line as possible from each piece
    Once we've created these lines, it guesses new lines using the approximate distance between all other lines 
    and places these in ranks 3-6 (inclusive). This is because, at the beginning of a game, 
    there will be no pieces on these ranks.

    Once it has 8 lines vertically and 8 lines horizontally, stretch each line to the edge of the board in order to 
    fill the any gaps that may occur. Find the intersection of each line and this gives an approximation 
    of each squares center point.
    """
    # Sort the points ao that we can match piece to its corresponding piece on the other side of the board
    left_points = bubbleSort(left_points, midpoints, False, 1)
    right_points = bubbleSort(right_points, midpoints, False, 1)
    top_points = bubbleSort(top_points, midpoints, False, 0)
    bot_points = bubbleSort(bot_points, midpoints, False, 0)
    horizontal = []
    vertical = []

    # Create the horizontal lines
    lines_img = cropped.copy()
    for i in range(len(left_points)):
        left_point = midpoints[left_points[i]]
        right_point = midpoints[right_points[i]]
        drawLine(lines_img, [left_point, right_point], (0, 255, 0), 5)
        line = [left_point, right_point]
        horizontal.append(line)

    # Create the vertical lines
    for i in range(len(top_points)):
        top_point = midpoints[top_points[i]]
        bot_point = midpoints[bot_points[i]]
        drawLine(lines_img, [top_point, bot_point], (0, 255, 0), 5)
        line = [top_point, bot_point]
        vertical.append(line)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'lines'), lines_img)

    # Depending on the orientation of the board, we either want to fill in lines horizontally or vertically
    if horizontalOrVertical(orientation) == 1:
        checker = horizontal
    else:
        checker = vertical

    act_vertical = []
    act_horizontal = []
    avg_line_distance = 0
    counter = 0
    # Get the average distance of each point in the respective lines list (checker)
    for i in range(0, len(checker) - 1):
        l1 = checker[i]
        l2 = checker[i + 1]
        p1 = l1[0]
        p2 = l2[0]
        dist = distance(p1, p2)

        p1 = l1[1]
        p2 = l2[1]
        dist += distance(p1, p2)
        dist /= 2
        avg_line_distance += dist
        counter += 2
    avg_line_distance /= counter
    # Scale the average line distance to .9 of its size
    # # TODO FUTURE WORK
    # # TODO Figure out how to do this in respect to the image distortion and positions from the center of the camera
    # # TODO More than likely not in the scope of the final year project
    avg_line_distance *= 0.9

    # Actually create the new lines from 'guessing'
    guess_img = cropped.copy()
    if horizontalOrVertical(orientation) == 1:
        line = horizontal[1]
        line2 = horizontal[2]

        p1 = (line[0][0], line[0][1] + avg_line_distance)
        p2 = (line[1][0], line[1][1] + avg_line_distance)
        p3 = (line2[0][0], line2[0][1] - avg_line_distance)
        p4 = (line2[1][0], line2[1][1] - avg_line_distance)

        avg_line_distance *= 0.9
        p5 = (p1[0], p1[1] + avg_line_distance)
        p6 = (p2[0], p2[1] + avg_line_distance)
        p7 = (p3[0], p3[1] - avg_line_distance)
        p8 = (p4[0], p4[1] - avg_line_distance)
    else:
        line = vertical[1]
        line2 = vertical[2]

        p1 = (line[0][0] + avg_line_distance, line[0][1])
        p2 = (line[1][0] + avg_line_distance, line[1][1])
        p3 = (line2[0][0] - avg_line_distance, line2[0][1])
        p4 = (line2[1][0] - avg_line_distance, line2[1][1])

        avg_line_distance *= 0.9
        p5 = (p1[0] + avg_line_distance, p1[1])
        p6 = (p2[0] + avg_line_distance, p2[1])
        p7 = (p3[0] - avg_line_distance, p3[1])
        p8 = (p4[0] - avg_line_distance, p4[1])

    new_line = [p1, p2]
    new_line2 = [p3, p4]
    new_line3 = [p5, p6]
    new_line4 = [p7, p8]

    # Draw each new line
    drawLine(guess_img, line, (0, 255, 0), 5)
    drawLine(guess_img, line2, (0, 255, 0), 5)
    drawLine(guess_img, new_line, (0, 255, 0), 5)
    drawLine(guess_img, new_line2, (0, 255, 0), 5)
    drawLine(guess_img, new_line3, (0, 255, 0), 5)
    drawLine(guess_img, new_line4, (0, 255, 0), 5)

    # Append the lines to the corresponding array depending on board orientation
    if horizontalOrVertical(orientation) == 1:
        horizontal.append(new_line)
        horizontal.append(new_line2)
        horizontal.append(new_line3)
        horizontal.append(new_line4)
    else:
        vertical.append(new_line)
        vertical.append(new_line2)
        vertical.append(new_line3)
        vertical.append(new_line4)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'guess_lines'), guess_img)

    # Stretch each line each line by finding the point they intersect with the edge of the board
    fixed_lines_img = cropped.copy()
    for i in range(len(vertical)):
        line = [(0, 0), (width, 0)]
        start_point = line_intersection(line, vertical[i])
        line = [(0, height), (width, height)]
        end_point = line_intersection(line, vertical[i])
        act_vertical.append([start_point, end_point])

    for i in range(len(horizontal)):
        line = [(0, 0), (0, height)]
        start_point = line_intersection(line, horizontal[i])
        line = [(width, 0), (width, height)]
        end_point = line_intersection(line, horizontal[i])
        act_horizontal.append([start_point, end_point])

    # Draw the new lines
    for i in range(len(act_vertical)):
        line = act_vertical[i]
        drawLine(fixed_lines_img, line, (0, 255, 0), 5)
    for i in range(len(act_horizontal)):
        line = act_horizontal[i]
        drawLine(fixed_lines_img, line, (0, 255, 0), 5)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'lines_fixed'), fixed_lines_img)

    """
    INTERSECTIONS
    In order to find the center of each square, we find the intersection of every single line
    The best way to do this is to sort the lines by where A1 is. IE: Is it in top left or bottom right

    Once its sorted correctly, we can just iterate through the horizontal lines, 
    getting the intersection point with each of the vertical lines, and that will give us each row of square centers
    """

    # Sort the lines according to where a1 is
    # If a1 is in the top left
    sorted_lines_horizontal = []
    sorted_lines_vertical = []
    if a1 == 0:
        sorted_lines_horizontal = bubbleSortLines(horizontal, False)
        sorted_lines_vertical = bubbleSortLines(vertical, False)
    # If a1 is in the top right
    elif a1 == 1:
        sorted_lines_horizontal = bubbleSortLines(vertical, True)
        sorted_lines_vertical = bubbleSortLines(horizontal, False)
    # If a1 is in the bottom right
    elif a1 == 2:
        sorted_lines_horizontal = bubbleSortLines(horizontal, True)
        sorted_lines_vertical = bubbleSortLines(vertical, True)
    # If a1 is in the bottom left
    elif a1 == 3:
        sorted_lines_horizontal = bubbleSortLines(vertical, False)
        sorted_lines_vertical = bubbleSortLines(horizontal, True)

    # Iterate through the lines
    # Get the intersection points
    # append it to a list of intersections
    # append it to a row which will be appended to another array to create a multidimensional array
    int_img = cropped.copy()
    intersections = []
    board = []
    for h in sorted_lines_horizontal:
        row = []
        for v in sorted_lines_vertical:
            point = line_intersection(h, v)
            intersections.append(point)
            row.append(point)
            drawCircle(int_img, point, (0, 255, 0))
        board.append(row)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'intersections'), int_img)

    # Display chess coordinates (a1, b2, c7, etc) on an image
    int_map_img = cropped.copy()
    # a = 0, b = 1, c = 2, d = 3, etc
    for i in range(len(board)):
        for j in range(len(board[i])):
            if j == 0:
                square = "a"
            elif j == 1:
                square = "b"
            elif j == 2:
                square = "c"
            elif j == 3:
                square = "d"
            elif j == 4:
                square = "e"
            elif j == 5:
                square = "f"
            elif j == 6:
                square = "g"
            elif j == 7:
                square = "h"

            square += str(i + 1)
            drawText(int_map_img, board[j][i], square, (0, 255, 0), 45)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'intersections_mapped_' + str(a1)), int_map_img)


    """
    PIECE MAPPINGS
    
    Create the first multidimensional array that holds the initial starting position of the game
    Read in the next image
    Crop it
    Detect the pieces
    Filter the pieces
    Find where the pieces are in relation to the intersection points
    Find what pieces are by taking the previous step and comparing it against the previous board state
    Convert the new board state to a FEN string
    """
    board_pieces = []
    row = ['R', 'P', '-', '-', '-', '-', 'p', 'r']
    board_pieces.append(row)
    row = ['N', 'P', '-', '-', '-', '-', 'p', 'n']
    board_pieces.append(row)
    row = ['B', 'P', '-', '-', '-', '-', 'p', 'b']
    board_pieces.append(row)
    row = ['Q', 'P', '-', '-', '-', '-', 'p', 'q']
    board_pieces.append(row)
    row = ['K', 'P', '-', '-', '-', '-', 'p', 'k']
    board_pieces.append(row)
    row = ['B', 'P', '-', '-', '-', '-', 'p', 'b']
    board_pieces.append(row)
    row = ['N', 'P', '-', '-', '-', '-', 'p', 'n']
    board_pieces.append(row)
    row = ['R', 'P', '-', '-', '-', '-', 'p', 'r']
    board_pieces.append(row)

    int_piece_img = cropped.copy()
    for i in range(len(board)):
        for j in range(len(board[i])):
            drawText(int_piece_img, board[j][i], board_pieces[j][i], (0, 255, 100), 45)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'intersections_piece_' + str(a1)), int_piece_img)

    states = []
    # Convert the board state into a FEN string format
    FEN = convertToFEN(board_pieces)
    print(FEN)
    # Append the FEN to a list of FEN strings and specify what order in the list it should be.
    """ This list would be what is inserted into the database """
    states.append({"order": 0, "state": FEN})
    counter = 1
    previous = board_pieces
    # Iterate through each image that is in the list of files. The first image has already been removed above.
    for file in files:
        path = VIDEO_PATH + file
        image = cv2.imread(path)
        # Rotate image rotato times
        for i in range(rotato):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # Crop the image with the rectangle from the first frame
        cropped = doCrop(image, rectangle)
        im_height, width, channels = cropped.shape
        image_expanded = np.expand_dims(cropped, axis=0)

        # Run the detection on the image
        (boxes, scores, classes, num) = sess.run(
            inputs,
            feed_dict={image_tensor: image_expanded})

        # Filter the detections
        boxes, classes, scores = doFilter(cropped, 1, boxes, classes, scores)
        # Get all the middle points of the detections
        middles = getMiddles(width, im_height, boxes)
        # Find where the detections are in relation to the intersection points acquired earlier
        diction = compareAgainstBoard(middles, board)
        # Convert the above comparison to a multidimensional array where pieces are ? and empty squares are -
        mapping = convertToBoardMapping(diction)
        # Compare the above mapping to the previous board state mapping
        # Figure out what each piece is
        mapping = compareAgainstPreviousState(mapping, previous)
        # Print the pieces onto an image
        labels_img = cropped.copy()
        for i in range(len(board)):
            for j in range(len(board[i])):
                piece = mapping[j][i]
                color = (240, 240, 240)
                if piece.isupper():
                    color = (15, 15, 15)
                if piece is not '-':
                    drawText(labels_img, board[j][i], piece, color, 50)
        cv2.imwrite("{}/{}.jpg".format(OUT_PATH, str(counter)), labels_img)
        # Set the previous equal to the mapping so we know what the new previous board state is
        previous = mapping
        # Convert the mapping to a fen string
        FEN = convertToFEN(mapping)
        print(FEN)
        # Append the FEN string to the list
        states.append({"order": counter, 'state': FEN})
        counter += 1

    end = time.time()
    print("It took {} seconds to process this video".format(end - start))


# Source folder of the sequence, should be located in Demo/static/net/input
sequence = "sequence1/"

# Theres a for loop here to demonstrate with different orientations of the board.
# i will be passed into the function which tells the algorithm to rotate the image i times
# i = 0, rotate 0 times
# i = 1, rotate the image once
# etc
for i in range(1):
    beginVideoProcessing("./static/net/input/" + sequence, "./static/net/output/" + sequence + str(i) + '/', i)
