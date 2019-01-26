import cv2
import os

VIDEO_DIR = "Videos/"
FRAMES_DIR = "Frames/"

'''
    Enter the names of each video to split. 
    Format must be .mp4
    output is .jpg
'''
VIDEO_LIST = ['1', '2', '3']


def checkDIR(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def splitVideo(video_name):
    video_path = VIDEO_DIR + video_name + '.mp4'
    frame_dest = FRAMES_DIR + video_name

    checkDIR(FRAMES_DIR)
    checkDIR(frame_dest)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    num_reads = 0
    while success:
        if num_reads % 5 == 0:
            cv2.imwrite("%s/%d.jpg" % (frame_dest, count), image)  # save frame as JPEG file
            count += 1
        success, image = vidcap.read()
        num_reads += 1

    print('Total Frames: %d' % num_reads)
    print('Total Stored: %d' % count)


for video in VIDEO_LIST:
    splitVideo(video)
