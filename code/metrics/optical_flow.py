from __future__ import division

import cv2 as cv
import numpy as np

def evaluate(testList, gtList):
    msen = []
    pepn = []
    motion_vector_errors = []
    errors_pixels = []
    valid_pixels_list = []
    for test_image, gt_image in zip(testList, gtList):

        # The optical flow images
        img = cv.imread(test_image, cv.IMREAD_UNCHANGED)
        gt_img = cv.imread(gt_image, cv.IMREAD_UNCHANGED)

        assert img.shape == gt_img.shape

        # MSEN: Mean Square Error in Non-occluded areas
        # PEPN: Percentage of Erroneous Pixels in Non-occluded areas
        error_msen, error_pepn, motion_vector_error, error_pixels, valid_pixels = flow_errors_MSEN_PEPN(img, gt_img)
        msen.append(error_msen)
        pepn.append(error_pepn)
        motion_vector_errors.append(motion_vector_error)
        errors_pixels.append(error_pixels)
        valid_pixels_list.append(valid_pixels)

    return msen, pepn, motion_vector_errors, errors_pixels, valid_pixels_list


def flow_errors_MSEN_PEPN(img, gt_img):
    assert img.shape == gt_img.shape

    optical_flow, _ = read_flow_field(img)
    optical_flow_gt, valid_pixels_gt = read_flow_field(gt_img)
    num_valid_pixels_gt = np.count_nonzero(valid_pixels_gt)

    optical_flow_diff = optical_flow - optical_flow_gt
    optical_flow_se = np.square(optical_flow_diff)
    motion_vector_errors = np.sqrt(np.sum(optical_flow_se, axis=-1))
    msen = np.sum(motion_vector_errors[valid_pixels_gt]) / num_valid_pixels_gt  # Only considering valid pixels

    error_pixels = np.logical_and(
        motion_vector_errors > 3.0,
        valid_pixels_gt
    )
    pepn = (np.count_nonzero(error_pixels) / num_valid_pixels_gt) * 100

    return msen, pepn, motion_vector_errors, error_pixels, valid_pixels_gt


def read_flow_field(img):
    # BGR -> RGB
    img = img[:, :, ::-1]

    optical_flow = img[:, :, :2].astype(float)
    optical_flow -= 2**15
    optical_flow /= 64.0
    valid_pixels = img[:, :, 2] == 1.0

    return optical_flow, valid_pixels