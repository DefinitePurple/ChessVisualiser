import math
import os
import random
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


def getContours(img):
    _, cnt, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt) <= 0:
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
    height, width, channel = _frame.shape

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
    if topLeft[0] < point[0] < bottomRight[0] and topLeft[1] < point[1] < bottomRight[1]:
        return True
    return False


def getMiddles(im_width, im_height, boxes):
    midpoints = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        midpoint = ((xmin * im_width + xmax * im_width) / 2, (ymin * im_height + ymax * im_height) / 2)
        midpoints.append(midpoint)
        # cv2.circle(cropped, midpoint, 10, (0, 0, 255), -1)

    return midpoints


def getMiddle(im_width, im_height, box):
    ymin, xmin, ymax, xmax = box
    midpoint = ((xmin * im_width + xmax * im_width) / 2, (ymin * im_height + ymax * im_height) / 2)
    return midpoint


def doFilter(cropped, threshold, boxes, classes, scores):
    # Filter out all detections with a score less than the threshold
    # And find multiple detections on the same piece, take the highest, remove the rest
    squeezed_boxes = np.squeeze(boxes)
    squeezed_scores = np.squeeze(scores)
    squeezed_classes = np.squeeze(classes).astype(np.int32)
    im_width, im_height, channel = cropped.shape

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
    midpoints = getMiddles(im_width, im_height, squeezed_boxes)

    # Find all the pieces that have a center inside another bounding box
    # Check if its score is less that the other ones
    # If it is, mark it for removal
    to_remove = []
    for i in range(len(squeezed_boxes)):
        box = squeezed_boxes[i]
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
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
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def getCorners(im_width, im_height, boxes):
    topLeftP = (0, 0)
    topRightP = (im_width, 0)
    botRightP = (im_width, im_height)
    botLeftP = (0, im_height)

    # (Position in array, distance from respective points)
    topLeftClosest = (-1, im_height * 10)
    topRightClosest = (-1, im_height * 10)
    botRightClosest = (-1, im_height * 10)
    botLeftClosest = (-1, im_height * 10)

    for i in range(len(boxes)):
        box = boxes[i]
        midpoint = getMiddle(im_width, im_height, box)
        if distance(midpoint, topLeftP) < topLeftClosest[1]:
            topLeftClosest = (i, distance(midpoint, topLeftP))

        if distance(midpoint, topRightP) < topRightClosest[1]:
            topRightClosest = (i, distance(midpoint, topRightP))

        if distance(midpoint, botRightP) < botRightClosest[1]:
            botRightClosest = (i, distance(midpoint, botRightP))

        if distance(midpoint, botLeftP) < botLeftClosest[1]:
            botLeftClosest = (i, distance(midpoint, botLeftP))

    return [topLeftClosest[0], topRightClosest[0], botRightClosest[0], botLeftClosest[0]]


def drawText(img, point, message, fill=(0, 0, 0)):
    image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
    d = ImageDraw.Draw(image_pil)
    font = Font.truetype(font="arial.ttf", size=40)
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


def perpDistFromLine(line, point):
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


# def sort(list, midpoints, reverse=False):
#     for passnum in range(len(list) - 1, 0, -1):
#         for i in range(passnum):
#             p1 = midpoints[list[i]][1]
#             p2 = midpoints[list[i + 1]][1]
#             if p1 > p2:
#                 temp = list[i]
#                 list[i] = list[i + 1]
#                 list[i + 1] = temp
#
#     return list


def bubbleSort(list, midpoints, reverse=False, axis=0):
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


def bubbleSortLines(list, reverse=False, axis=0):
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
    mid = ((line[0][0] + line[1][1]) / 2, (line[1][0] + line[1][0]) / 2)
    return mid


def line_intersection(line1, line2):
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


def getOrientation(img, corners, boxes):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg_luminance = 0
    counter = 0
    im_width, im_height, _ = rgb.shape
    for mid in getMiddles(im_width, im_height, boxes):
        pixel = list(rgb[mid[1] - 1:mid[1], mid[0] - 1: mid[0]][0][0])
        avg_luminance += 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
        counter += 1
    avg_luminance /= counter

    orientation = []
    for corner in corners:
        pixel = list(rgb[corner[1] - 1:corner[1], corner[0] - 1: corner[0]][0][0])
        Y = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
        if Y >= avg_luminance:
            orientation.append(1)
        else:
            orientation.append(0)

    return orientation


def horizontalOrVertical(orientation):
    if orientation == [1, 0, 0, 1] or orientation == [0, 1, 1, 0]:
        return 0  # Vertical
    elif orientation == [0, 0, 1, 1] or orientation == [1, 1, 0, 0]:
        return 1  # Horizontal
    else:
        return -1  # Error


# Returns position in corners_mids array of which corner is 'a1'
# Starts clockwise from top left
def whereIsA1(orientation):
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


def sortFiles(files, reverse=False):
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


def compareAgainstBoard(middles, board):
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
    print(diction)
    return diction


def beyondFirstFrame(img, inputs, sess, category_index, image_tensor, board):
    # Detect pieces
    # Filter
    # Get middles
    # Check against the board

    im_height, im_width, channels = img.shape
    image_expanded = np.expand_dims(img, axis=0)
    (boxes, scores, classes, num) = sess.run(
        inputs,
        feed_dict={image_tensor: image_expanded})

    boxes, classes, scores = doFilter(img, 1, boxes, classes, scores)
    labels_img = img.copy()
    # visualization_utils.visualize_boxes_and_labels_on_image_array(
    #     labels_img,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=3,
    #     min_score_thresh=0.01,
    #     max_boxes_to_draw=50)
    # cv2.imshow('frame', labels_img)
    # cv2.waitKey(100000)
    middles = getMiddles(im_width, im_height, boxes)
    diction = compareAgainstBoard(middles, board)
    for p in diction:
        point = board[p[0]][p[1]]
        drawText(labels_img, point, str(p[0]) + str(p[1]))
        drawCircle(labels_img, point)
    cv2.imwrite("./static/net/output/sequence/{}.jpg".format('tellytubbies' + str(random.randint(0, 1000))), labels_img)


def beginVideoProcessing(VIDEO_PATH, OUT_PATH):
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

    files = os.listdir(VIDEO_PATH)
    files = sortFiles(files, False)
    path = VIDEO_PATH + files[0]
    img = cv2.imread(path)
    rectangle = crop(img)
    cropped = doCrop(img, rectangle)

    test_lines = cropped.copy()
    image_expanded = np.expand_dims(cropped, axis=0)
    im_width, im_height, channel = cropped.shape

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

    """
    FILTER
    Filter each bounding box to take the highest scoring of each overlapping box
    If the center point of a box is inside another box but its score is smaller, remove it  
    """
    boxes, classes, scores = doFilter(cropped, 1, boxes, classes, scores)
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

    visualization_utils.visualize_boxes_and_labels_on_image_array(
        test_lines,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.01,
        max_boxes_to_draw=50)

    """
    CORNERS
    Get the array index of each corner point
    Get the center point of each corner
    Append the center point to a list to hold the corner mid points    
    """
    corners = getCorners(im_width, im_height, boxes)
    corners_mids = []
    for i in corners:
        tmp = getMiddle(im_width, im_height, boxes[i])
        drawCircle(test_lines, tmp, (0,0,0), 10)
        corners_mids.append(tmp)

    """
    ORIENTATION
    Convert image to RGB
    Get the luminance at the center of every piece and calculate the average luminance
    Check if the luminance of each corner piece is higher or lower than the average
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
    midpoints = getMiddles(im_width, im_height, boxes)
    # This is the distance to compare against when checking the distance from a line
    check = distance((0, 0), (im_width / 20, im_height / 20))
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

    middles = getMiddles(im_width, im_height, boxes)
    for mid in middles:
        print(mid)
        drawCircle(test_lines, mid, (255, 255, 255), 20)
    # Sort the points ao that we can match piece to its corresponding piece on the other side of the board
    left_points = bubbleSort(left_points, midpoints, False, 1)
    right_points = bubbleSort(right_points, midpoints, False, 1)
    top_points = bubbleSort(top_points, midpoints, False, 0)
    bot_points = bubbleSort(bot_points, midpoints, False, 0)
    horizontal = []
    vertical = []

    # Create the horizontal lines
    for i in range(len(left_points)):
        left_point = midpoints[left_points[i]]
        right_point = midpoints[right_points[i]]
        line = [left_point, right_point]
        horizontal.append(line)

    # Create the vertical lines
    for i in range(len(top_points)):
        top_point = midpoints[top_points[i]]
        bot_point = midpoints[bot_points[i]]
        line = [top_point, bot_point]
        vertical.append(line)

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
    # # TODO Figure out how to do this in respect to the image distortion and positions from the center of the camera
    # # TODO More than likely not in the scope of the final year project
    avg_line_distance *= 0.9

    # Actually create the new lines from 'guessing'
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

    # Append the lines to the corresponding array depending on board orientation
    # if horizontalOrVertical(orientation) == 1:
    #     horizontal.append(new_line)
    #     horizontal.append(new_line2)
    #     horizontal.append(new_line3)
    #     horizontal.append(new_line4)
    # else:
    #     vertical.append(new_line)
    #     vertical.append(new_line2)
    #     vertical.append(new_line3)
    #     vertical.append(new_line4)

    # Stretch each line each line by finding the point they intersect with the edge of the board
    for i in range(len(vertical)):
        line = [(0, 0), (im_width, 0)]
        start_point = line_intersection(line, vertical[i])
        line = [(0, im_height), (im_width, im_height)]
        end_point = line_intersection(line, vertical[i])
        act_vertical.append([start_point, end_point])

    for i in range(len(horizontal)):
        line = [(0, 0), (0, im_height)]
        start_point = line_intersection(line, horizontal[i])
        line = [(im_width, 0), (im_width, im_height)]
        end_point = line_intersection(line, horizontal[i])
        act_horizontal.append([start_point, end_point])

    # Sort the lines according to where a1 is
    # If a1 is in the top left
    if a1 == 0:
        sorted_lines_horizontal = bubbleSortLines(horizontal)
        sorted_lines_vertical = bubbleSortLines(vertical, False, 1)
    # If a1 is in the top right
    elif a1 == 1:
        sorted_lines_horizontal = bubbleSortLines(horizontal, False)
        sorted_lines_vertical = bubbleSortLines(vertical, True, 1)
    # If a1 is in the bottom right
    elif a1 == 2:
        sorted_lines_horizontal = bubbleSortLines(horizontal, True)
        sorted_lines_vertical = bubbleSortLines(vertical, True, 1)
    # If a1 is in the bottom left
    elif a1 == 3:
        sorted_lines_horizontal = bubbleSortLines(horizontal, True)
        sorted_lines_vertical = bubbleSortLines(vertical, False, 1)

    intersections = []
    board = []
    for h in sorted_lines_horizontal:
        row = []
        for v in sorted_lines_vertical:
            point = line_intersection(h, v)
            intersections.append(point)
            drawCircle(labels_img, point, (255, 0, 0), 4)
            row.append(point)
        board.append(row)


    for line in sorted_lines_horizontal:
        drawLine(test_lines, line)
    for line in sorted_lines_vertical:
        drawLine(test_lines, line)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'test_lines'), test_lines)

    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'Yes'), labels_img)
    test_img = cropped.copy()
    # a = 0, b = 1, c = 2, d = 3, etc
    drawText(test_img, board[3][2], "c4")
    drawText(test_img, board[1][0], "a2")
    drawText(test_img, board[2][7], "h3")
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'test'), test_img)

    files.pop(0)
    frames = []

    for file in files:
        path = VIDEO_PATH + file
        image = cv2.imread(path)
        frames.append(beyondFirstFrame(doCrop(image, rectangle), inputs, sess, category_index, image_tensor, board))
    end = time.time()
    print("It took {} seconds to process this video".format(end - start))


# beginVideoProcessing("../Data/Images/Dataset/ByPiece/20190313_112101.jpg", "./static/net/output")

beginVideoProcessing("./static/net/input/sequence/", "./static/net/output/sequence")
