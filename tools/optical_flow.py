from __future__ import division

import itertools
import logging

import numpy as np
import time
import cv2 as cv

import sys
import tqdm

from scipy import stats
from matplotlib import pyplot as plt

def exhaustive_search_block_matching(reference_img, search_img, block_size=16, max_search_range=16, norm='l1',
                                     verbose=False):

    logger = logging.getLogger(__name__)

    norm_options = {'l1', 'l2'}
    start_time = time.time()
    height = reference_img.shape[0]
    width = reference_img.shape[1]

    # Assertions
    assert block_size > 0, 'Block size should be bigger than 0 pixels'
    assert max_search_range > 0, 'Max search range should be bigger than 0 pixels'
    assert norm in norm_options, '{} norm not supported. Choose one of {}'.format(norm, norm_options)

    # Pad reference image to have dimensions multiple of block size
    pad_ref_height = int(block_size * np.ceil(height / block_size))
    pad_ref_width = int(block_size * np.ceil(width / block_size))
    pad_y = pad_ref_height - height
    pad_x = pad_ref_width - width

    pad_reference_img = np.pad(reference_img, ((0, pad_y), (0, pad_x)), mode='constant')

    # Placeholder for predicted frame and optical flow
    pad_predicted_frame = np.empty_like(pad_reference_img, dtype=np.uint8)
    num_blocks_width, num_blocks_height = int(pad_ref_width / block_size), int(pad_ref_height / block_size)
    optical_flow = np.zeros((num_blocks_height, num_blocks_width, 2))
    total_number_blocks = num_blocks_width * num_blocks_height

    # Loop through every NxN block in the target image
    for (block_row, block_col) in tqdm.tqdm(itertools.product(
            range(0, pad_ref_height - (block_size - 1), block_size),
            range(0, pad_ref_width - (block_size - 1), block_size)
    ), desc='Exhaustive Block Matching progress', total=total_number_blocks, file=sys.stdout):

        # Current block in the reference image
        block = pad_reference_img[block_row:block_row + block_size, block_col:block_col + block_size]

        # Placeholders for minimum norm and matching block
        dfd_n_min = np.infty
        matching_block = np.zeros((block_size, block_size))

        # Search in a surronding region, determined by search_range
        search_range = range(-max_search_range, block_size + max_search_range)
        for (search_col, search_row) in itertools.product(search_range, search_range):
            # Up left corner of the candidate block
            up_left_y = block_row + search_row
            up_left_x = block_col + search_col
            # Bottom right corner of the candidate block
            bottom_right_y = block_row + search_row + block_size - 1
            bottom_right_x = block_col + search_col + block_size - 1

            # Do not search if upper left corner is defined outside the reference image
            if up_left_y < 0 or up_left_x < 0:
                continue
            # Do not search if bottom right corner is defined outside the reference image
            if bottom_right_y >= height or bottom_right_x >= width:
                continue

            # Get the candidate block
            candidate_block = search_img[up_left_y:bottom_right_y+1, up_left_x:bottom_right_x+1]
            assert candidate_block.shape == (block_size, block_size)

            # Compute the Displaced Frame Difference (DFD) and compute the specified norm
            dfd = np.array(candidate_block, dtype=np.float32) - np.array(block, dtype=np.float32)
            norm_order = 2 if norm == 'l2' else 1
            candidate_dfd_norm = np.linalg.norm(dfd, ord=norm_order)

            # Store the minimum norm and corresponding displacement vector
            if candidate_dfd_norm < dfd_n_min:
                dfd_n_min = candidate_dfd_norm
                matching_block = candidate_block
                dy = search_col
                dx = search_row

        # construct the predicted image with the block that matches this block
        pad_predicted_frame[block_row:block_row + block_size, block_col:block_col + block_size] = matching_block

        if verbose:
            logger.info(
                "Block [{blk_row}, {blk_col}] out of [{total_blks_rows}, {total_blks_cols}] --> "
                "Displacement: ({dx}, {dy})\t Minimum DFD norm: {norm}".format(
                    blk_row=block_row // block_size,
                    blk_col=block_col // block_size,
                    total_blks_rows=num_blocks_height,
                    total_blks_cols=num_blocks_width,
                    dx=dx,
                    dy=dy,
                    norm=dfd_n_min,
                )
            )

        # Store displacement of this block in each direction
        optical_flow[block_row // block_size, block_col // block_size, 0] = dx
        optical_flow[block_row // block_size, block_col // block_size, 1] = dy

    # Create dense optical flow to match input image dimensions by repeating values
    dense_optical_flow = np.repeat(optical_flow, block_size, axis=0)
    dense_optical_flow = np.repeat(dense_optical_flow, block_size, axis=1)

    end_time = time.time()
    total_time = end_time - start_time

    logger.info('Total time: {:.2f} s\tTime per block: {:.2f} s'.format(
        total_time, total_time / (num_blocks_height * num_blocks_width)
    ))

    # Crop results to match real input dimensions
    if pad_y != 0 and pad_x != 0:
        predicted_frame = pad_predicted_frame[:-pad_y, :-pad_x]
        dense_optical_flow = dense_optical_flow[:-pad_y, :-pad_x]
    elif pad_y != 0:
        predicted_frame = pad_predicted_frame[:-pad_y, :]
        dense_optical_flow = dense_optical_flow[:-pad_y, :]
    elif pad_x != 0:
        predicted_frame = pad_predicted_frame[:, :-pad_x]
        dense_optical_flow = dense_optical_flow[:, :-pad_x]
    else:
        predicted_frame = pad_predicted_frame
        dense_optical_flow = dense_optical_flow

    return predicted_frame, optical_flow, dense_optical_flow, total_time


def opencv_optflow(ref_img_data, search_img_data, block_size):
    farneback_params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': block_size,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': cv.OPTFLOW_USE_INITIAL_FLOW
    }
    dense_flow = cv.calcOpticalFlowFarneback(ref_img_data, search_img_data, **farneback_params)

    return dense_flow


def video_stabilization(image, flow, direction, u, v):

    mean_u = int(np.round(stats.trim_mean(flow[:, :, 0], 0.2, axis=None)))
    mean_v = int(np.round(stats.trim_mean(flow[:, :, 1], 0.2, axis=None)))

    print('Displacement: (%s,%s)' % (mean_u, mean_v))

    if direction == 'forward':
        mean_u = u - mean_u
        mean_v = v - mean_v
    else:
        mean_u = u + mean_u
        mean_v = v + mean_v

    rect_image = np.zeros(image.shape)
    if mean_u == 0 and mean_v == 0:
        rect_image = image
    elif mean_u == 0:
        if mean_v > 0:
            rect_image[:, mean_v:, :] = image[:, :-mean_v, :]
        else:
            rect_image[:, :mean_v, :] = image[:, -mean_v:, :]
    elif mean_v == 0:
        if mean_u > 0:
            rect_image[mean_u:, :, :] = image[:-mean_u, :, :]
        else:
            rect_image[:mean_u, :, :] = image[-mean_u:, :, :]
    elif mean_u > 0:
        if mean_v > 0:
            rect_image[mean_u:, mean_v:, :] = image[:-mean_u, :-mean_v, :]
        else:
            rect_image[mean_u:, :mean_v, :] = image[:-mean_u, -mean_v:, :]
    else:
        if mean_v > 0:
            rect_image[:mean_u, mean_v:, :] = image[-mean_u:, :-mean_v, :]
        else:
            rect_image[:mean_u, :mean_v, :] = image[-mean_u:, -mean_v:, :]

    return rect_image, mean_u, mean_v
