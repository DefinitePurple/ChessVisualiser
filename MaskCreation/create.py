"""
    NOTE: DANIEL FITZPATRICK C15345046 IS NOT THE AUTHOR OF THE MAJORITY OF THIS CODE.
    IT HAS BEEN SOURCED FROM http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/
"""

from PIL import Image  # (pip install Pillow)
import numpy as np  # (pip install numpy)
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
import json
import cv2
from matplotlib import pyplot as plt

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation


def getMask(lower, upper, img):
    lower = np.array(lower)
    upper = np.array(upper)
    return cv2.inRange(img, lower, upper)


# Read image
source = cv2.imread('test.jpg')
"""
    THE FOLLOWING IS CREATED BY DANIEL FITZPATRICK AS A SOLUTION FOR IMPERFECTIONS IN JPG IMAGE FORMAT
"""
red_mask = getMask([210, 0, 0], [255, 40, 40], source)
blue_mask = getMask([0, 0, 210], [40, 40, 255], source)
green_mask = getMask([0, 210, 0], [40, 255, 40], source)
ROI = red_mask + blue_mask + green_mask

black_mask = getMask([0, 0, 0], [30, 30, 30], source)

chess_image_mask = source.copy()  # Copy the source image
chess_image_mask[np.where(red_mask != 0)] = (0, 0, 255)
chess_image_mask[np.where(green_mask != 0)] = (0, 255, 0)
chess_image_mask[np.where(blue_mask != 0)] = (255, 0, 0)
chess_image_mask [np.where(black_mask != 0)] = (0, 0, 0)

plt.imshow(chess_image_mask)
plt.show()

""" 
    THE FOLLOWING HAS BEEN EDITED BY DANIEL FITZPATRICK IN ORDER TO REFLECT THE PROJECT REQUIREMENTS
"""
mask_images = [chess_image_mask]
# Define which colors match which categories in the images
black_piece_id, white_piece_id = [1, 2]
category_ids = {
    1: {
        '(255, 0, 0)': black_piece_id,
        '(0, 0, 255)': black_piece_id,
        '(0, 255, 0)': white_piece_id
    }
}

is_crowd = 0

# These ids will be automatically increased as we go
annotation_id = 1
image_id = 1

# Create the annotations
annotations = []
for mask_image in mask_images:
    sub_masks = create_sub_masks(mask_image)
    for color, sub_mask in sub_masks.items():
        category_id = category_ids[image_id][color]
        annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
        annotations.append(annotation)
        annotation_id += 1
    image_id += 1

print(json.dumps(annotations))
