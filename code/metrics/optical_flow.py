import logging
import cv2 as cv
import numpy as np

def evaluate(testList, gtList):
    logger = logging.getLogger(__name__)
    for test_image, gt_image in zip(testList, gtList):
        img = cv.imread(test_image, cv.CV_LOAD_IMAGE_COLOR)
        gt_img = cv.imread(gt_image, cv.CV_LOAD_IMAGE_COLOR)
        h = len(img)
        w = len(img[0])
        # Check if background interpolation is needed
        if h == len(gt_img) and w == len(gt_img[0]):
            interpolate = False

        # Access BGR intensity values
        for i in range(0, h):
            for j in range(0, w):
                intensity = img[i, j]
                blue = intensity[0]
                green = intensity[1]
                red = intensity[2]

#def set_flow_U(img, row, column, value):