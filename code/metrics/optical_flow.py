import logging
import cv2 as cv
import numpy as np
import math

def evaluate(testList, gtList):
    logger = logging.getLogger(__name__)

    msen = []
    pepn = []
    for test_image, gt_image in zip(testList, gtList):

        # The optical flow images
        img = cv.imread(test_image, cv.CV_LOAD_IMAGE_COLOR)
        gt_img = cv.imread(gt_image, cv.CV_LOAD_IMAGE_COLOR)

        h = len(img)
        w = len(img[0])
        # Check if background interpolation is needed
        if h == len(gt_img) and w == len(gt_img[0]):
            interpolate = False
        else:
            interpolate = True

        # MSEN: Mean Square Error in Non-occluded areas
        error_msen = flow_error_MSEN(img, gt_img)
        msen.append(error_msen)

        # PEPN: Percentage of Erroneous Pixels in Non-occluded areas
        # error_pepn = flow_errors_PEPN(img, gt_img)
        # pepn.append(error_pepn)

    return msen, pepn

# MSEN: Mean Square Error in Non-occluded areas
def flow_error_MSEN(img, gt_img):
    h = len(img)
    w = len(img[0])
    # Check if background interpolation is needed
    if h != len(gt_img) and w != len(gt_img[0]):
        print('Error: image sizes does not match')
        return

    error = 0
    num_pixels = 0
    # Access all pixels
    for row in range(0, h):
        for col in range(0, w):
            fu = gt_img[row, col][0] - img[row, col][0]
            fv = gt_img[row, col][1] - img[row, col][1]
            f_error = math.sqrt(fu * fu + fv * fv);
            if f_error > 3:
                error += f_error
                num_pixels += 1

    # Normalize error
    error = error / max(num_pixels, 1)

    return error

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
            f_error = math.sqrt(fu*fu + fv*fv);
            if gt_img[row, col][2] == 1:
                errors[0] += f_error
                num_pixels += 1
                if img[row, col][2] == 1:
                    errors[1] += f_error
                    num_pixels_result +=1

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
            f_error = math.sqrt(fu*fu + fv*fv);
            if gt_img[row, col][2] == 1:
                for i in range(0, 5):
                    if f_error > i+1:
                        errors[i*2] +=1
                num_pixels += 1
                if img[row, col][2] == 1:
                    for i in range(0, 5):
                        if f_error > i + 1:
                            errors[i*2 + 1] += 1
                    num_pixels_result +=1

    # Check number of pixels
    if num_pixels == 0:
        print('Error: ground truth defect')
        return

    # Normalize errors
    errors = errors / max(num_pixels, 1)
    if num_pixels_result > 0:
        errors /= max(num_pixels_result, 1)

    # Density
    density = num_pixels_result / max(num_pixels,1)
    errors.append(density)
    return errors

# Method in Kitti C++ io_flow.h
# Method to construct flow image from PNG file
def read_flow_field(img):

    h = len(img)
    w = len(img[0])
    optical_flow = np.empty((h, w, 3))
    # optical_flow_image = np.zeros((h, w, 3), np.uint8)

    # Access BGR intensity values
    for i in range(0, h):
        for j in range(0, w):
            blue = img[i, j][0]
            green = img[i, j][1]
            red = img[i, j][2]
            if blue > 0:
                # Set optical flow u-component at given value
                optical_flow[i, j][0] = (red - 32768.0) / 64.0
                # Set optical flow v-component at given value
                optical_flow[i, j][1] = (green - 32768.0) / 64.0
                # Set optical flow at given pixel to valid/invalid
                optical_flow[i, j][2] = 1

    return optical_flow