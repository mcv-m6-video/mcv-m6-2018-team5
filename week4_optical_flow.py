#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os
import itertools
import pickle

import cv2 as cv
import numpy as np

from config.load_configutation import Configuration
from metrics import optical_flow
from tools import visualization, optical_flow
from tools.image_parser import get_image_list_kitti_dataset
from tools.log import log_context
from tools.mkdirs import mkdirs

EPSILON = 1e-8

def optical_flow(cf):
    with log_context(cf.log_file):

        logger = logging.getLogger(__name__)

        logger.info(' ---> Init test: ' + cf.test_name + ' <---')

        if cf.dataset_name == 'kitti':

            # Get a list with input images filenames
            image_list = get_image_list_kitti_dataset(cf.dataset_path, cf.image_sequences, cf.image_type)

            # Get a list with groung truth images filenames
            gt_list = get_image_list_kitti_dataset(cf.gt_path, cf.image_sequences, cf.image_type)


            # Task 1 Block Matching Algorithm

            optical_flow.bloch_matching_algorithm()


            # Get a list with test results filenames
            test_list = get_image_list_kitti_dataset(cf.results_path, cf.image_sequences, cf.image_type, 'LKflow_')

            if cf.evaluate:
                # Call the method to evaluate the optical flow
                msen, pepn, squared_errors, pixel_errors, valid_pixels = optical_flow.evaluate(test_list, gt_list)
                logger.info('Mean Squared Error: {}'.format(msen))
                logger.info('Percentage of Erroneous Pixels: {}'.format(pepn))

                for mse, se, pe, vp, seq_name, original_image in zip(msen, squared_errors, pixel_errors, valid_pixels,
                                                                     cf.image_sequences, image_list):
                    # Histogram
                    visualization.plot_histogram_msen(mse, np.ravel(se[vp]), seq_name, cf.output_folder)
                    # Image
                    visualization.plot_msen_image(original_image, se, pe, vp, seq_name, cf.output_folder)

            if cf.plot_optical_flow:
                for image, test_image, seq_name in zip(image_list, test_list, cf.image_sequences):
                    # Quiver plot
                    output_path = os.path.join(cf.output_folder, 'optical_flow_{}.png'.format(seq_name))
                    visualization.plot_optical_flow(image, test_image, cf.optical_flow_downsample, seq_name, output_path)

                    # HSV plot
                    output_path = os.path.join(cf.output_folder, 'optical_flow_hsv_{}.png'.format(seq_name))
                    visualization.plot_optical_flow_hsv(image, test_image, seq_name, output_path)

        logger.info(' ---> Finish test: ' + cf.test_name + ' <---')


# Main function
def main():
    # Task choices
    tasks = {
        'optical_flow': optical_flow
    }

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Video surveillance application, Team 5')
    parser.add_argument('task', choices=tasks.keys(), help='Task to run')
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
    task_fn = tasks[arguments.task]
    task_fn(cf)


# Entry point of the script
if __name__ == "__main__":
    main()