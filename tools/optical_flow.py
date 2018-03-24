from __future__ import division

import itertools
import logging

import numpy as np
import time
import cv2 as cv

import sys
import tqdm

from scipy import stats

from tools.others import stabFuntions


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
            candidate_block = search_img[up_left_y:bottom_right_y + 1, up_left_x:bottom_right_x + 1]
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
    if '3' in cv.__version__:
        dense_flow = cv.calcOpticalFlowFarneback(ref_img_data, search_img_data, None, **farneback_params)
    elif '2.4' in cv.__version__:
        dense_flow = cv.calcOpticalFlowFarneback(ref_img_data, search_img_data, **farneback_params)
    else:
        dense_flow = None

    return dense_flow


def video_stabilization(image, flow, optical_flow_mode, strategy, area_search, acc_direction, previous_direction,
                        running_avg_weight, **kwargs):
    # Unpack directions
    previous_u, previous_v = previous_direction
    acc_u, acc_v = acc_direction

    # Find compensation optical_flow_mode
    if strategy == 'max':
        xedges = np.arange(-area_search, area_search + 2)
        yedges = np.arange(-area_search, area_search + 2)
        flow_hist2d, _, _ = np.histogram2d(
            np.ravel(flow[:, :, 0]), np.ravel(flow[:, :, 1]), bins=(xedges, yedges)
        )
        flow_hist2d = flow_hist2d.T

        max_pos = np.argwhere(flow_hist2d == flow_hist2d.max())
        y_max_pos = max_pos[0, 0]
        x_max_pos = max_pos[0, 1]
        u_compensate = xedges[x_max_pos]
        v_compensate = yedges[y_max_pos]
    elif strategy == 'trimmed_mean':
        # Use the mean optical_flow_mode to compensate
        u_compensate = int(np.round(stats.trim_mean(flow[:, :, 0], 0.2, axis=None)))
        v_compensate = int(np.round(stats.trim_mean(flow[:, :, 1], 0.2, axis=None)))

    elif strategy == 'background_blocks':
        center_positions = kwargs['center_positions']
        neighborhood = kwargs['neighborhood']
        u_compensate = 0
        v_compensate = 0
        for center_i, center_j in center_positions:
            u_compensate_vals = flow[center_i-neighborhood:center_i+neighborhood,
                                     center_j-neighborhood:center_j+neighborhood, 0]
            v_compensate_vals = flow[center_i-neighborhood:center_i+neighborhood,
                                     center_j-neighborhood:center_j+neighborhood, 1]
            u_compensate += np.mean(u_compensate_vals)
            v_compensate += np.mean(v_compensate_vals)
        u_compensate /= len(center_positions)
        v_compensate /= len(center_positions)

    else:
        raise ValueError('Strategy {!r} not supported. Use one of: [max, trimmed_mean, background_blocks]'.format(
            strategy
        ))

    print('Displacement (before running avg.): (%s,%s)' % (u_compensate, v_compensate))

    # Compute a running average
    u_compensate = running_avg_weight * previous_u + (1 - running_avg_weight) * u_compensate
    v_compensate = running_avg_weight * previous_v + (1 - running_avg_weight) * v_compensate

    print('Displacement (after running avg.): (%s,%s)' % (u_compensate, v_compensate))

    if optical_flow_mode == 'forward':
        acc_u = int(acc_u - u_compensate)
        acc_v = int(acc_v - v_compensate)
    else:
        acc_u = int(acc_u + u_compensate)
        acc_v = int(acc_v + v_compensate)

    print('Accumulated displacement: (%s,%s)' % (acc_u, acc_v))

    rect_image = np.zeros(image.shape)
    if acc_u == 0 and acc_v == 0:
        rect_image = image
    elif acc_u == 0:
        if acc_v > 0:
            rect_image[:, acc_v:, :] = image[:, :-acc_v, :]
        else:
            rect_image[:, :acc_v, :] = image[:, -acc_v:, :]
    elif acc_v == 0:
        if acc_u > 0:
            rect_image[acc_u:, :, :] = image[:-acc_u, :, :]
        else:
            rect_image[:acc_u, :, :] = image[-acc_u:, :, :]
    elif acc_u > 0:
        if acc_v > 0:
            rect_image[acc_u:, acc_v:, :] = image[:-acc_u, :-acc_v, :]
        else:
            rect_image[acc_u:, :acc_v, :] = image[:-acc_u, -acc_v:, :]
    else:
        if acc_v > 0:
            rect_image[:acc_u, acc_v:, :] = image[-acc_u:, :-acc_v, :]
        else:
            rect_image[:acc_u, :acc_v, :] = image[-acc_u:, -acc_v:, :]

    return rect_image, (acc_u, acc_v), (u_compensate, v_compensate)


def video_stabilization_sota(prev_gray, cur_gray, prev_to_cur_transform, prev_corner):
    cur_corner, status, err = cv.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_corner, None)
    # storage for keypoints with status 1
    prev_corner2 = []
    cur_corner2 = []
    for i, st in enumerate(status):
        # if keypoint found in frame i & i-1
        if st == 1:
            # store coords of keypoints that appear in both
            prev_corner2.append(prev_corner[i])
            cur_corner2.append(cur_corner[i])
    prev_corner2 = np.array(prev_corner2)
    prev_corner2 = np.array(prev_corner2)
    cur_corner2 = np.array(cur_corner2)
    # estimate partial transform (resource: http://nghiaho.com/?p=2208)
    T_new = cv.estimateRigidTransform(prev_corner2, cur_corner2, False)
    if T_new is not None:
        T = T_new
    # translation x
    dx = T[0, 2]
    # translation y
    dy = T[1, 2]
    # rotation
    da = np.arctan2(T[1, 0], T[0, 0])
    # store for saving to disk as table
    prev_to_cur_transform.append([dx, dy, da])

    return prev_to_cur_transform


def video_stabilization_sota2(videoInList, videoOutPath):
    # detector and matcher
    detector = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # parameters
    MATCH_THRES = float('Inf')
    RANSAC_THRES = 0.2
    BORDER_CUT = 10
    # FILT = "square"
    FILT = "gauss"
    FILT_WIDTH = 7
    FILT_SIGMA = 0.2
    FAST = True
    if FILT == "square":
        filt = (1.0 / FILT_WIDTH) * np.ones(FILT_WIDTH)
        suffix = "_MT_" + str(MATCH_THRES) + "_RT_" + str(RANSAC_THRES) + "_FILT_" + FILT + "_FW_" + str(
            FILT_WIDTH) + "_FAST_" + str(FAST)
    elif FILT == "gauss":
        filtx = np.linspace(-3 * FILT_SIGMA, 3 * FILT_SIGMA, FILT_WIDTH)
        filt = np.exp(-np.square(filtx) / (2 * FILT_SIGMA))
        filt = 1 / (np.sum(filt)) * filt
        suffix = "_MT_" + str(MATCH_THRES) + "_RT_" + str(RANSAC_THRES) + "_FILT_" + FILT + "_FW_" + str(
            FILT_WIDTH) + "_SG_" + str(FILT_SIGMA) + "_FAST_" + str(FAST)

    # numpy array
    frame = cv.imread(videoInList[0])
    videoArr = np.zeros((len(videoInList), frame.shape[0], frame.shape[1], frame.shape[2]), dtype=np.uint8)
    # fill array
    for i in range(0, len(videoInList)):
        videoArr[i, :, :] = cv.imread(videoInList[i])

    ### get transformation
    trans = stabFuntions.getTrans(videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt, FAST)
    # plotTrans(trans, None, videoBaseName, suffix, show=False)

    # video reconstruction
    stabFuntions.reconVideo(videoInList, videoOutPath, trans, BORDER_CUT)


def read_flo_flow(name):
    flow = None
    with open(name, 'rb') as f:
        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def read_kitti_flow(img):
    # BGR -> RGB
    img = img[:, :, ::-1]

    optical_flow = img[:, :, :2].astype(float)
    optical_flow -= 2 ** 15
    optical_flow /= 64.0
    valid_pixels = img[:, :, 2] == 1.0

    return optical_flow, valid_pixels