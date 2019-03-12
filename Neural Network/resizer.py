# Use to resize the dataset from original size to 0.3 of its size
# Useful because image straight from phone could be, for example, 2k x 4k pixels
# Which is waaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaay too large and means itd take ages to train our model
# However, will try it on the large set later ;)

import cv2
import os

dir_path = os.getcwd()

for filename in os.listdir(dir_path):
    if filename.endswith(".jpg"):
        print(filename)
        image = cv2.imread(filename)
        resized = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        cv2.imwrite(filename, resized)
