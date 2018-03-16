from __future__ import division

import logging

import cv2 as cv
import numpy as np
import time
import sys

def block_matching_algorithm(cf, current_image, previous_image):
    logger = logging.getLogger(__name__)

    start = time.time()

    # Read Input images in GRAY
    c_img = cv.imread(current_image, cv.IMREAD_GRAYSCALE)
    p_img = cv.imread(previous_image, cv.IMREAD_GRAYSCALE)

    # Block Matching algorithm
    if cf.compensation == 'backward':
        input_image = c_img
        search_image = p_img
    else:
        input_image = p_img
        search_image = c_img

    img_height = input_image.shape[0]
    img_width = input_image.shape[1]

    # Reshape to a multiple of blocksize
    h_tmp = img_height // cf.block_size
    w_tmp = img_width // cf.block_size
    input_image = input_image[:h_tmp*cf.block_size,:w_tmp*cf.block_size]

    img_height = input_image.shape[0]
    img_width = input_image.shape[1]

    search_image = search_image[:img_height,:img_width]
    # Get the block size
    block_h = img_height // cf.block_size
    block_w = img_width // cf.block_size

    # Add padding in the search image
    pad_search_image = np.zeros([img_height + 2 * cf.search_area, img_width + 2 * cf.search_area])
    pad_search_image[cf.search_area : cf.search_area + img_height, cf.search_area : cf.search_area + img_width] = search_image[:,:]

    motion_matrix = np.zeros([block_h, block_w, 2])

    for row in range(block_h):
        for col in range(block_w):

            search_block = input_image[row * cf.block_size:row * cf.block_size + cf.block_size,
                              col * cf.block_size:col * cf.block_size + cf.block_size]
            region = pad_search_image[row * cf.block_size:row * cf.block_size + cf.block_size + 2 * cf.search_area,
                                col * cf.block_size:col * cf.block_size + cf.block_size + 2 * cf.search_area]

            h_motion, w_motion = match_block(region, search_block, cf.block_size, cf.search_area)

            motion_matrix[row, col, 0] = h_motion
            motion_matrix[row, col, 1] = w_motion

    end = time.time()

    logger.info("Block matching estimated in {:.2f} s".format(end - start))

    return motion_matrix


def match_block(region, block_to_search, block_size, search_area):

    h_size = region.shape[0]
    w_size = region.shape[1]

    min_diff = sys.float_info.max

    for row in range(h_size - search_area):
        for col in range(w_size - search_area):
            region_block = region[row : row + block_size, col : col + block_size]
            diff = sum(sum(abs(region_block - block_to_search)**2))
            if diff < min_diff:
                min_diff = diff
                h_motion = - row + search_area
                w_motion = col - search_area

    return h_motion, w_motion


def expand_motion_matrix(motion_matrix, block_size):

    pixel_motion_matrix=np.zeros([motion_matrix.shape[0]*block_size,motion_matrix.shape[1]*block_size,2])

    for xx in range(motion_matrix.shape[0]):
        for yy in range(motion_matrix.shape[1]):
            pixel_motion_matrix[xx*block_size:xx*block_size+block_size,yy*block_size:yy*block_size+block_size,0] = motion_matrix[xx,yy,0]
            pixel_motion_matrix[xx*block_size:xx*block_size+block_size,yy*block_size:yy*block_size+block_size,1] = motion_matrix[xx,yy,1]

    return pixel_motion_matrix