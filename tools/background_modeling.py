from __future__ import division

import logging

import cv2 as cv
import numpy as np
import time


def single_gaussian_modelling(back_list):
    logger = logging.getLogger(__name__)

    start = time.time()

    back_img = cv.imread(back_list[0], cv.IMREAD_GRAYSCALE)
    mean = np.zeros((back_img.shape[0], back_img.shape[1]))
    variance = np.zeros((back_img.shape[0], back_img.shape[1]))

    for back_image in back_list:
        back_img = cv.imread(back_image, cv.IMREAD_GRAYSCALE)
        mean = mean + back_img
    mean = mean / len(back_list)

    for back_image in back_list:
        back_img = cv.imread(back_image, cv.IMREAD_GRAYSCALE)
        variance = variance + np.square(back_img - mean)
    variance = variance / len(back_list)

    end = time.time()

    logger.info("Background estimated in {:.2f} s".format(end - start))

    return mean, variance


def multivariative_gaussian_modelling(back_list, color_space="RGB"):
    logger = logging.getLogger(__name__)

    start = time.time()

    back_img = cv.imread(back_list[0])
    if color_space == "HSV":
        back_img = cv.cvtColor(back_img, cv.COLOR_BGR2HSV)
        mean = np.zeros((back_img.shape[0], back_img.shape[1], 2))
        variance = np.zeros((back_img.shape[0], back_img.shape[1], 2))
    else:
        mean = np.zeros((back_img.shape[0], back_img.shape[1], back_img.shape[2]))
        variance = np.zeros((back_img.shape[0], back_img.shape[1], back_img.shape[2]))

    for back_image in back_list:
        back_img = cv.imread(back_image)
        if color_space == "HSV":
            back_img = cv.cvtColor(back_img, cv.COLOR_BGR2HSV)

        mean[:, :, 0] = mean[:, :, 0] + back_img[:, :, 0]
        mean[:, :, 1] = mean[:, :, 1] + back_img[:, :, 1]
        if color_space == "RGB":
            mean[:, :, 2] = mean[:, :, 2] + back_img[:, :, 2]

    mean = mean / len(back_list)

    for back_image in back_list:
        back_img = cv.imread(back_image)
        if color_space == "HSV":
            back_img = cv.cvtColor(back_img, cv.COLOR_BGR2HSV)
        variance[:, :, 0] = variance[:, :, 0] + np.square(back_img[:, :, 0] - mean[:, :, 0])
        variance[:, :, 1] = variance[:, :, 1] + np.square(back_img[:, :, 1] - mean[:, :, 1])
        if color_space == "RGB":
            variance[:, :, 2] = variance[:, :, 2] + np.square(back_img[:, :, 2] - mean[:, :, 2])

    variance = variance / len(back_list)

    end = time.time()

    logger.info("Background estimated in {:.2f} s".format(end - start))

    return mean, variance


def foreground_estimation(img, mean, variance, alpha):
    threshold = alpha * (np.sqrt(variance) + 2)
    img = np.abs(cv.imread(img, cv.IMREAD_GRAYSCALE) - mean)
    foreground = (img >= threshold)
    return foreground


def foreground_estimation_color(img, mean, variance, alpha, color_space):
    img = cv.imread(img)
    if color_space == "HSV":
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = np.abs(img[:,:,:-1] - mean)
    else:
        img = np.abs(img - mean)
    threshold = alpha * (np.sqrt(variance) + 2)
    if color_space == "RGB":
        foreground = (img[:, :, 0] >= threshold[:, :, 0]) | \
                     (img[:, :, 1] >= threshold[:, :, 1]) | \
                     (img[:, :, 2] >= threshold[:, :, 2])
    else:
        foreground = (img[:, :, 0] >= threshold[:, :, 0]) | \
                     (img[:, :, 1] >= threshold[:, :, 1])
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

    variance = (rho * np.square(img_background - mean) + (1 - rho) * variance) * back + variance * (1 - back)
    return foreground, mean, variance


def adaptive_foreground_estimation_color(img, mean, variance, alpha, rho, color_space):
    img = cv.imread(img)
    if color_space == "HSV":
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = img[:, :, 0:1]

    img_norm = np.abs(img - mean)
    threshold = alpha * (np.sqrt(variance) + 2)
    if color_space == "RGB":
        foreground = (img_norm[:, :, 0] >= threshold[:, :, 0]) | \
                     (img_norm[:, :, 1] >= threshold[:, :, 1]) | \
                     (img_norm[:, :, 2] >= threshold[:, :, 2])
    else:
        foreground = (img_norm[:, :, 0] >= threshold[:, :, 0]) | \
                     (img_norm[:, :, 1] >= threshold[:, :, 1])
    foreground = np.expand_dims(foreground, axis=-1)
    back = (1 - foreground)
    img_background = img * back
    # The mean for the Background pixels (back) is adapted, the mean for the foreground pixels remains the same
    mean = (rho * img_background + (1 - rho) * mean) * back + mean * (1 - back)
    # The variance for the Background pixels (back) is adapted, the variance for the foreground pixels remains the same

    variance = (rho * np.square(img_background - mean) + (1 - rho) * variance) * back + variance * (1 - back)

    foreground = np.take(foreground, indices=0, axis=-1)
    return foreground, mean, variance
