#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os
from skimage import io as skio
import cv2 as cv

from tools import background_modeling, foreground_improving, detection, multi_tracking, visualization, \
    image_rectification
from tools.image_parser import get_image_list_changedetection_dataset
from utils.load_configutation import Configuration
from utils.log import log_context
from utils.mkdirs import mkdirs

EPSILON = 1e-8


# noinspection PyUnboundLocalVariable
def traffic_monitoring(cf):
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
        H, shape = image_rectification.rectify_image(mean)

        # Instantiate tracker for multi-object tracking
        #tracker = Tracker(cf.distance_threshold, cf.max_frames_to_skip, cf.max_trace_length, 0, cf)

        tracks = []  # Create an empty array of tracks.

        nextId = 1  # ID of the next track

        for image_path in foreground_img_list:

            foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                image_path, mean, variance, cf.alpha, cf.rho, cf.color_space)

            foreground = foreground_improving.hole_filling(foreground, cf.four_connectivity)
            foreground = foreground_improving.remove_small_regions(foreground, cf.area_filtering_P)
            foreground = foreground_improving.image_opening(foreground, cf.opening_strel, cf.opening_strel_size)
            foreground = foreground_improving.image_closing(foreground, cf.closing_strel, cf.closing_strel_size)

            image = skio.imread(image_path, as_grey=True)
            image = image_rectification.wrap(image, H, shape)
            foreground = image_rectification.wrap(foreground, H, shape)

            bboxes, centroids = detection.detectObjects(image, foreground)
            multi_tracking.predictNewLocationsOfTracks(tracks)
            assignments, unassignedTracks, unassignedDetections = multi_tracking.detectionToTrackAssignment(tracks, centroids, cf.costOfNonAssignment)
            multi_tracking.updateAssignedTracks(tracks, bboxes, centroids, assignments)
            multi_tracking.updateUnassignedTracks(tracks, unassignedTracks)
            multi_tracking.deleteLostTracks(tracks)
            multi_tracking.createNewTracks(tracks, bboxes, centroids, unassignedDetections)

            if cf.save_results:
                image_name = os.path.basename(image_path)
                image_name = os.path.splitext(image_name)[0]
                save_path = os.path.join(cf.results_path, image_name + '.' + cf.result_image_type)
                image = cv.imread(image_path)
                image = image_rectification.wrap(image, H, shape)
                image = (255 * image).astype('uint8')
                visualization.displayTrackingResults(image, tracks, foreground, save_path)


        logger.info(' ---> Finish test: ' + cf.test_name + ' <---')


# Main function
def main():

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Traffic monitoring [Team 5]')
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
    traffic_monitoring(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
