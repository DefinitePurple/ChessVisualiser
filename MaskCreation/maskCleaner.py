import cv2
import os
from matplotlib import pyplot as plt
import glob

DATA_DIR = '../Data'
IMAGES_DIR = '/Images'
imgs = os.listdir(DATA_DIR + IMAGES_DIR)
imgs = [k for k in imgs if 'masks' in k and '.jpg' in k and '~' not in k]

height = width = 600
dim = (width, height)

for img_path in imgs:
    full_path = DATA_DIR + IMAGES_DIR + '/' + img_path
    print(full_path)
    img = cv2.imread(full_path)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    for i in range(0, 600):
        for j in range(0, 600):
            color = resized[i][j]
            print(color)
            if list(resized[i][j]) is [0, 0, 0]:
                print('next')
            break
        break

    plt.imshow(resized)
    plt.show()
    # cv2.imwrite(DATA_DIR + IMAGES_DIR + '/COPY-' + img, resized)
