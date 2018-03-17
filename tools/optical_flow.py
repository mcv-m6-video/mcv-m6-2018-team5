from __future__ import division

import itertools
import logging

import numpy as np
import time


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

    # Placeholder for predicted frame and optical flow
    predicted_frame = np.empty_like(reference_img, dtype=np.uint8)
    num_blocks_width, num_blocks_height = int(width / block_size), int(height / block_size)
    optical_flow = np.zeros((num_blocks_height, num_blocks_width, 2))

    # Loop through every NxN block in the target image
    for (block_row, block_col) in itertools.product(
            range(0, height - (block_size - 1), block_size),
            range(0, width - (block_size - 1), block_size)
    ):

        # Current block in the reference image
        block = reference_img[block_row:block_row + block_size, block_col:block_col + block_size]

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
        predicted_frame[block_row:block_row + block_size, block_col:block_col + block_size] = matching_block

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
        optical_flow[block_row // block_size, block_col // block_size, 0] = dy
        optical_flow[block_row // block_size, block_col // block_size, 1] = dx

    # Create dense optical flow to match input image dimensions by repeating values
    dense_optical_flow = np.repeat(optical_flow, block_size, axis=0)
    dense_optical_flow = np.repeat(dense_optical_flow, block_size, axis=1)

    end_time = time.time()
    total_time = end_time - start_time

    logger.info('Total time: {:.0f} s\tTime per block: {:.0f} s'.format(
        total_time, total_time / (num_blocks_height * num_blocks_width)
    ))

    return predicted_frame, optical_flow, dense_optical_flow
