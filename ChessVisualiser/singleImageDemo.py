import math
import time
import os
import tensorflow as tf
import sys
import cv2
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
# Import utilites
from utils_demo import label_map_util
from utils_demo import visualization_utils


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
        midpoint = (int((xmin * im_width + xmax * im_width) / 2), int((ymin * im_height + ymax * im_height) / 2))
        midpoints.append(midpoint)
        # cv2.circle(cropped, midpoint, 10, (0, 0, 255), -1)

    return midpoints


def getMiddle(im_width, im_height, box):
    ymin, xmin, ymax, xmax = box
    midpoint = (int((xmin * im_width + xmax * im_width) / 2), int((ymin * im_height + ymax * im_height) / 2))
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


# DONT USE, Only rotate the array dont need to care about image
def rotate(img, orientation):
    if orientation == [0, 1, 1, 0]:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == [1, 0, 0, 1]:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == [1, 1, 0, 0]:
        return cv2.rotate(img, cv2.ROTATE_180)


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


def sortVeritcally(list, midpoints):
    for passnum in range(len(list) - 1, 0, -1):
        for i in range(passnum):
            p1 = midpoints[list[i]][1]
            p2 = midpoints[list[i + 1]][1]
            if p1 > p2:
                temp = list[i]
                list[i] = list[i + 1]
                list[i + 1] = temp

    return list


def sortHorizontally(list, midpoints):
    for passnum in range(len(list) - 1, 0, -1):
        for i in range(passnum):
            p1 = midpoints[list[i]][0]
            p2 = midpoints[list[i + 1]][0]
            if p1 > p2:
                temp = list[i]
                list[i] = list[i + 1]
                list[i + 1] = temp

    return list


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


def beginVideoProcessing(VIDEO_PATH, OUT_PATH):
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

    img = cv2.imread(VIDEO_PATH)
    rectangle = crop(img)
    cropped = doCrop(img, rectangle)

    image_expanded = np.expand_dims(cropped, axis=0)
    im_width, im_height, channel = cropped.shape

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
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

    middles = getMiddles(im_width, im_height, boxes)
    middles_img = cropped.copy()
    for mid in middles:
        drawCircle(middles_img, mid, (0, 255, 0), 10)
    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'middles'), middles_img)

    corners = getCorners(im_width, im_height, boxes)
    corner_img = cropped.copy()
    corners_mids = []
    for i in corners:
        tmp = getMiddle(im_width, im_height, boxes[i])
        corners_mids.append(tmp)
        drawCircle(corner_img, tmp, (0, 255, 0), 10)

    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'corners'), corner_img)

    # Orientation
    # Convert image to grayscale
    # Check if each pixel on the centerpoint of the corner pieces
    # is closer to either black or white.
    # white is 1, black is 0

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    orientation = []
    for corner in corners_mids:
        pixel = gray[corner[1] - 1:corner[1], corner[0] - 1: corner[0]][0]
        dist = 255 - pixel
        if dist < pixel:
            orientation.append(1)
        else:
            orientation.append(0)


    top_points = []
    right_points = []
    bot_points = []
    left_points = []

    top_points.append(corners[0])
    top_points.append(corners[1])
    right_points.append(corners[1])
    right_points.append(corners[2])
    bot_points.append(corners[2])
    bot_points.append(corners[3])
    left_points.append(corners[3])
    left_points.append(corners[0])

    midpoints = getMiddles(im_width, im_height, boxes)
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

    left_points = sortVeritcally(left_points, midpoints)
    right_points = sortVeritcally(right_points, midpoints)
    top_points = sortHorizontally(top_points, midpoints)
    bot_points = sortHorizontally(bot_points, midpoints)
    horizontal = []
    vertical = []

    lines_img = cropped.copy()
    for i in range(len(left_points)):
        left_point = midpoints[left_points[i]]
        right_point = midpoints[right_points[i]]
        drawLine(lines_img, [left_point, right_point], (0, 255, 0), 5)
        line = [left_point, right_point]
        horizontal.append(line)

    for i in range(len(top_points)):
        top_point = midpoints[top_points[i]]
        bot_point = midpoints[bot_points[i]]
        drawLine(lines_img, [top_point, bot_point], (0, 255, 0), 5)
        line = [top_point, bot_point]
        vertical.append(line)

    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'lines'), lines_img)

    act_vertical = []
    act_horizontal = []

    avg_line_distance = 0
    counter = 0

    for i in range(0, len(horizontal) - 1):
        l1 = horizontal[i]
        l2 = horizontal[i + 1]
        p1 = l1[0]
        p2 = l2[0]
        dist = distance(p1, p2)

        p1 = l1[1]
        p2 = l2[1]
        dist += distance(p1, p2)
        dist /= 2
        avg_line_distance += dist
        counter += 1
    avg_line_distance /= counter

    fixed_lines_img = cropped.copy()
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

    for i in range(len(act_vertical)):
        line = act_vertical[i]
        drawLine(fixed_lines_img, line, (0, 255, 0), 5)

    for i in range(len(act_horizontal)):
        line = act_horizontal[i]
        drawLine(fixed_lines_img, line, (0, 255, 0), 5)

    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'lines_fixed'), fixed_lines_img)

    # Check the colors of each corner piece
    # Find all pieces closest to bottom
    # Find all pieces closest to the top
    # Find all pieces closest to left
    # Find all pieces closest to right
    # Create lines connecting each piece
    # Find pieces positions from each line by checking their verticality and horizontalness


    # drawCircle(cropped, (top, left), 10, (255, 0, 0))

    """
        Create an array to hold vertical lines where the ends of the lines are the pieces on rank 1 and 8
    """

    # for i in range(len(scores)):
    #     print(category_index[classes[i]]['name'], scores[i])

    cv2.imwrite("{}/{}.jpg".format(OUT_PATH, 'cropped'), cropped)

    # print(np.squeeze(scores))
    end = time.time()
    print("It took {} seconds to process this video".format(end - start))


# beginVideoProcessing("../Data/Images/Dataset/ByPiece/20190313_112101.jpg", "./static/net/output")

beginVideoProcessing("./static/net/input/2.jpg", "./static/net/output")
