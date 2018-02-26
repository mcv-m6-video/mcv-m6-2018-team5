from __future__ import division

import cv2 as cv
import numpy as np


def evaluate(testList, gtList):
    msen = []
    pepn = []
    for test_image, gt_image in zip(testList, gtList):

        # The optical flow images
        img = cv.imread(test_image, cv.IMREAD_COLOR)
        gt_img = cv.imread(gt_image, cv.IMREAD_COLOR)

        assert img.shape == gt_img.shape

        # MSEN: Mean Square Error in Non-occluded areas
        # PEPN: Percentage of Erroneous Pixels in Non-occluded areas
        error_msen, error_pepn = flow_errors_MSEN_PEPN(img, gt_img)
        msen.append(error_msen)
        pepn.append(error_pepn)

    return msen, pepn


def flow_errors_MSEN_PEPN(img, gt_img):
    optical_flow, _ = read_flow_field(img)
    optical_flow_gt, valid_pixels_gt = read_flow_field(gt_img)

    optical_flow_se = np.square(optical_flow - optical_flow_gt)
    motion_vector_errors = np.sqrt(np.sum(optical_flow_se, axis=-1))

    error_pixels = np.logical_and(
        motion_vector_errors > 3.0,
        valid_pixels_gt
    )

    num_valid_pixels = np.count_nonzero(valid_pixels_gt)

    msen = np.sum(optical_flow_se[valid_pixels_gt]) / num_valid_pixels
    pepn = np.count_nonzero(error_pixels) / num_valid_pixels

    return msen, pepn


def read_flow_field(img):
    # BGR -> RGB
    img = img[:, :, ::-1]

    h, w, c = img.shape
    optical_flow = np.zeros((h, w, 2), dtype=float)

    optical_flow[:, :, 0] = (img[:, :, 0] - 2**15) / 64.0
    optical_flow[:, :, 1] = (img[:, :, 1] - 2**15) / 64.0
    # TODO: Check why none of the pixels in the last channel are different than 0
    valid_pixels = np.ones((h, w), dtype=bool)

    return optical_flow, valid_pixels


"""
# Method in Kitti C++ evaluate_flow.cpp
def flow_errors_outlier(img, gt_img):
    h = len(img)
    w = len(img[0])
    # Check if background interpolation is needed
    if h != len(gt_img) and w != len(gt_img[0]):
        print('Error: image sizes does not match')
        return

    errors = np.empty(2)
    num_pixels = 0
    num_pixels_result = 0

    # Access all pixels
    for row in range(0, h):
        for col in range(0, w):
            fu = gt_img[row, col][0] - img[row, col][0]
            fv = gt_img[row, col][1] - img[row, col][1]
            f_error = math.sqrt(fu * fu + fv * fv);
            if gt_img[row, col][2] == 1:
                errors[0] += f_error
                num_pixels += 1
                if img[row, col][2] == 1:
                    errors[1] += f_error
                    num_pixels_result += 1

    # Normalize errors
    errors[0] = errors[0] / max(num_pixels, 1)
    errors[1] = errors[1] / max(num_pixels_result, 1)

    return errors


# Method in Kitti C++ evaluate_flow.cpp
def flow_errors_average(img, gt_img):
    h = len(img)
    w = len(img[0])
    # Check if background interpolation is needed
    if h != len(gt_img) and w != len(gt_img[0]):
        print('Error: image sizes does not match')
        return

    errors = np.empty(10)
    num_pixels = 0
    num_pixels_result = 0

    # Access all pixels
    for row in range(0, h):
        for col in range(0, w):
            fu = gt_img[row, col][0] - img[row, col][0]
            fv = gt_img[row, col][1] - img[row, col][1]
            f_error = math.sqrt(fu * fu + fv * fv);
            if gt_img[row, col][2] == 1:
                for i in range(0, 5):
                    if f_error > i + 1:
                        errors[i * 2] += 1
                num_pixels += 1
                if img[row, col][2] == 1:
                    for i in range(0, 5):
                        if f_error > i + 1:
                            errors[i * 2 + 1] += 1
                    num_pixels_result += 1

    # Check number of pixels
    if num_pixels == 0:
        print('Error: ground truth defect')
        return

    # Normalize errors
    errors = errors / max(num_pixels, 1)
    if num_pixels_result > 0:
        errors /= max(num_pixels_result, 1)

    # Density
    density = num_pixels_result / max(num_pixels, 1)
    errors.append(density)
    return errors
"""
