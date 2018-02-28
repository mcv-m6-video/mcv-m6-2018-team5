from __future__ import division

import cv2 as cv
import numpy as np
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


def plot_optical_flow(img_path, vector_field_path, downsample_factor, sequence_name, output_path):

    # Get the original image
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # Get the optical flow image
    optical_flow, _ = read_flow_field(cv.imread(vector_field_path, cv.IMREAD_UNCHANGED))

    # Downsample optical flow image
    optical_flow_ds = block_reduce(optical_flow, block_size=(downsample_factor, downsample_factor, 1), func=np.mean)
    x_pos = np.arange(0, img.shape[1], downsample_factor)
    y_pos = np.arange(0, img.shape[0], downsample_factor)
    X = np.meshgrid(x_pos)
    Y = np.meshgrid(y_pos)

    plt.imshow(img, cmap='gray')
    plt.quiver(X, Y, optical_flow_ds[:, :, 0], optical_flow_ds[:, :, 1], color='yellow')
    plt.axis('off')
    plt.title(sequence_name)
    plt.show(block=False)
    plt.savefig(output_path)
    plt.close()

def plot_optical_flow_hsv(img_path, vector_field_path, sequence_name, output_path):

    # Get the original image
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # Get the optical flow image
    optical_flow, _ = read_flow_field(cv.imread(vector_field_path, cv.IMREAD_UNCHANGED))

    magnitude, angle = cv.cartToPolar(np.square(optical_flow[:, :, 0]), np.square(optical_flow[:, :, 1]),
                                      None, None, True)
    magnitude = cv.normalize(magnitude, 0, 255, norm_type=cv.NORM_MINMAX)

    optical_flow_hsv = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    optical_flow_hsv[..., 0] = angle
    optical_flow_hsv[..., 1] = 255
    optical_flow_hsv[:, :, 2] = magnitude
    # optical_flow_hsv[:, :, 1] = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    optical_flow_rgb = cv.cvtColor(optical_flow_hsv, cv.COLOR_HSV2BGR)

    plt.imshow(img, cmap='gray')
    plt.imshow(optical_flow_rgb, alpha=0.5)
    plt.axis('off')
    plt.title(sequence_name)
    plt.show(block=False)
    plt.savefig(output_path)
    plt.close()

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
