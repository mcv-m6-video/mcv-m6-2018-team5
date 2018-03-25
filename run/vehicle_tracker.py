#! /usr/bin/env python

from __future__ import division

import argparse
import itertools
import logging
import os
import pickle
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from utils.load_configutation import Configuration
from tools.metrics import optical_flow as of_metrics
from tools import optical_flow as of
from tools import visualization
from tools.image_parser import get_sequence_list_kitti_dataset, get_gt_list_kitti_dataset, \
    get_image_list_changedetection_dataset, get_image_list_ski_video_dataset
from utils.log import log_context
from utils.mkdirs import mkdirs

EPSILON = 1e-8


# noinspection PyUnboundLocalVariable
def vehicle_tracker(cf):
    with log_context(cf.log_file):
        logger = logging.getLogger(__name__)
        logger.info(' ---> Init test: ' + cf.test_name + ' <---')



        logger.info(' ---> Finish test: ' + cf.test_name + ' <---')


# Main function
def main():

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='W5 - Vehicle tracker and speed estimator [Team 5]')
    parser.add_argument('-c', '--config-path', type=str, required=True, help='Configuration file path')
    parser.add_argument('-t', '--test-name', type=str, required=True, help='Name of the test')

    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration' \
                                              'path using -c config/path/name' \
                                              ' in the command line'
    assert arguments.test_name is not None, 'Please provide a name for the ' \
                                            'test using -e test_name in the ' \
                                            'command line'

    # Load the configuration file
    configuration = Configuration(arguments.config_path, arguments.test_name)
    cf = configuration.load()

    # Run task
    vehicle_tracker(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
