import math
import os
import time

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import cv2
import numpy as np
import tensorflow as tf

import ChessVisualiser.emailHandler as emailer
# Import utilites
from ChessVisualiser.utils import label_map_util
from ChessVisualiser.utils import visualization_utils as vis_util


def getContours(img):
    _, cnt, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    boundary = 50
    # Set the corners
    height, width, channel = _frame.shape
    topLeftX = rectangle[0] - boundary
    topLeftY = rectangle[1] - boundary
    bottomRightX = rectangle[0] + rectangle[2] + boundary
    bottomRightY = rectangle[1] + rectangle[3] + boundary

    if topLeftX < 0:
        topLeftX = 0

    if topLeftY < 0:
        topLeftY = 0

    if bottomRightX > width:
        bottomRightX = width

    if bottomRightY > height:
        bottomRightY = height

    topLeft = (topLeftX, topLeftY)
    bottomRight = (bottomRightX, bottomRightY)

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


def drawCircle(img, point, radius=5, color=(255, 0, 0)):
    x = point[0]
    y = point[1]
    image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    np.copyto(img, np.array(image_pil))


def drawLine(img, line, color=(255, 0, 0)):
    image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    draw.line(line, color, 3)
    np.copyto(img, np.array(image_pil))


def getOrientation(img, corners):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orientation = []
    for corner in corners:
        pixel = gray[corner[1] - 1:corner[1], corner[0] - 1: corner[0]][0]
        dist = 255 - pixel
        if dist < pixel:
            orientation.append(1)
        else:
            orientation.append(0)
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
            # cv2.imwrite("{}/{}.jpg".format(PATH, int(pos_frame)), cropped)
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

    # print("Frames to keep {}".format(framesToKeepPos))
    cap.release()
    video.release()
    end = time.time()
    print("It took {} seconds to process this video".format(end - start))
    emailer.sendEmail('processed', user['email'], {'username': user['username']})

# beginVideoProcessing("./static/users/1/videos/24/20190323145923.mp4")
