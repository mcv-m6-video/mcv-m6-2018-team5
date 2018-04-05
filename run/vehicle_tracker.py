#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os

import cv2 as cv
from skimage import io as skio

from tools import background_modeling, foreground_improving, visualization, detection
from tools.image_parser import get_image_list_changedetection_dataset
from tools.multi_tracking import MultiTracker
from utils.load_configutation import Configuration
from utils.log import log_context
from utils.mkdirs import mkdirs

EPSILON = 1e-8


# noinspection PyUnboundLocalVariable
def vehicle_tracker(cf):
    with log_context(cf.log_file):
        logger = logging.getLogger(__name__)
        logger.info(' ---> Init test: ' + cf.test_name + ' <---')

        if cf.save_results:
            logger.info('Saving results in {}'.format(cf.results_path))
            mkdirs(cf.results_path)

        image_list = get_image_list_changedetection_dataset(
            cf.dataset_path, 'in', cf.first_image, cf.image_type, cf.nr_images
        )

        background_img_list = image_list[:len(image_list) // 2]
        foreground_img_list = image_list[(len(image_list) // 2):]
        mean, variance = background_modeling.multivariative_gaussian_modelling(background_img_list, cf.color_space)

        # Instantiate tracker for multi-object tracking
        multi_tracker = MultiTracker(cf.costOfNonAssignment)

        for image_path in foreground_img_list:

            foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                image_path, mean, variance, cf.alpha, cf.rho, cf.color_space)

            foreground = foreground_improving.hole_filling(foreground, cf.four_connectivity)
            foreground = foreground_improving.remove_small_regions(foreground, cf.area_filtering_P)
            foreground = foreground_improving.image_opening(foreground, cf.opening_strel, cf.opening_strel_size)
            foreground = foreground_improving.image_closing(foreground, cf.closing_strel, cf.closing_strel_size)

            image = skio.imread(image_path, as_grey=True)
            bboxes, centroids = detection.detectObjects(image, foreground)

            # Tracking
            multi_tracker.predict_new_locations_of_tracks()
            multi_tracker.detection_to_track_assignment(centroids)
            multi_tracker.update_assigned_tracks(bboxes, centroids)
            multi_tracker.update_unassigned_tracks()
            multi_tracker.delete_lost_tracks()
            multi_tracker.create_new_tracks(bboxes, centroids)

            if cf.save_results:
                image_name = os.path.basename(image_path)
                image_name = os.path.splitext(image_name)[0]
                save_path = os.path.join(cf.results_path, image_name + '.' + cf.result_image_type)
                image = cv.imread(image_path)
                visualization.displayTrackingResults(image, multi_tracker.tracks, foreground, save_path)

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
