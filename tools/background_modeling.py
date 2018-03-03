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


def foreground_estimation(img, mean, variance, alpha):
    threshold = alpha * (np.sqrt(variance) + 2)
    img = np.abs(cv.imread(img, cv.IMREAD_GRAYSCALE) - mean)
    foreground = (img >= threshold)
    return foreground

def adaptive_foreground_estimation(img, mean, variance, alpha, rho):
    threshold = alpha * (np.sqrt(variance) + 2)
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    img_norm = np.abs(img - mean)
    foreground = (img_norm >= threshold)
    back = (1 - foreground)
    img_background = img * back
    # The mean for the Background pixels (back) is adapted, the mean for the foreground pixels remains the same
    mean = (rho * img_background + (1 - rho) * mean) * back + mean * (1 - back)
    # The variance for the Background pixels (back) is adapted, the variance for the foreground pixels remains the same

    variance = (rho * np.square(img_background - mean) + (1-rho)*(variance))*back + (variance)*(1-back)
    return foreground, mean, variance

def mog_foreground_estimation(img, fgbg):
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    foreground = fgbg.apply(img)
    return foreground, fgbg

def mog2_foreground_estimation(img, fgbg):
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    foreground = fgbg.apply(img)
    return foreground, fgbg

def gmg_foreground_estimation(img, fgbg):
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    foreground = fgbg.apply(img)
    return foreground, fgbg

def lsbp_foreground_estimation(img, fgbg):
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    foreground = fgbg.apply(img)
    return foreground, fgbg

