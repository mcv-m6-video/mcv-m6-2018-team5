from __future__ import division

import logging

import cv2 as cv
import numpy as np
import time


def bloch_matching_algorithm():
    logger = logging.getLogger(__name__)

    start = time.time()

    # Block Matching algorithm

    end = time.time()

    logger.info("Optical flow estimated in {:.2f} s".format(end - start))

    # return optical_flow