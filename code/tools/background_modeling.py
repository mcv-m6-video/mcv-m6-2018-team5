
import logging

import cv2 as cv
import numpy as np
import time

def single_gaussian_modelling(backList):
    logger = logging.getLogger(__name__)

    start = time.time()

    back_img = cv.imread(backList[0], cv.IMREAD_GRAYSCALE)
    mean = np.zeros((back_img.shape[0], back_img.shape[1]))
    variance = np.zeros((back_img.shape[0], back_img.shape[1]))

    for back_image in backList:
        back_img = cv.imread(back_image, cv.IMREAD_GRAYSCALE)
        mean = mean + back_img
    mean = mean / len(backList)

    for back_image in backList:
        back_img = cv.imread(back_image, cv.IMREAD_GRAYSCALE)
        variance = variance + np.square(back_img - mean)
    variance = variance / len(backList)

    end = time.time()

    logger.info("Background estimated in {:.2f} s".format(end - start))

    return mean, variance

def foreground_estimation(imageList, mean, variance, alpha):

    foregrounds = []
    threshold = alpha * (np.sqrt(variance) + 2)
    for image in imageList:
        img = np.abs(cv.imread(image, cv.IMREAD_GRAYSCALE) - mean)
        fore = (img >= threshold)
        foregrounds.append(fore)

    return foregrounds






