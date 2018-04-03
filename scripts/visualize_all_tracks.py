#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os

from skimage import io as skio
from skimage import measure as skmeasure
from skimage import morphology as skmorph
import cv2 as cv
import numpy as np

from tools import background_modeling, foreground_improving, visualization, multi_tracking, detection
from tools.image_parser import get_image_list_changedetection_dataset
from tools.tracking import Tracker
from utils.load_configutation import Configuration
from utils.log import log_context
from utils.mkdirs import mkdirs

EPSILON = 1e-8
CAR_COLOURS = [(0, 0, 255), (0, 106, 255), (0, 216, 255), (0, 255, 182), (0, 255, 76),
               (144, 255, 0), (255, 255, 0), (255, 148, 0), (255, 0, 178), (220, 0, 255)]

# noinspection PyUnboundLocalVariable
def visualize_all_tracks(cf):
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

            bboxes, centroids = detection.detectObjects(image_path, foreground)
            multi_tracking.predictNewLocationsOfTracks(tracks)
            assignments, unassignedTracks, unassignedDetections = multi_tracking.detectionToTrackAssignment(tracks, centroids, cf.costOfNonAssignment)
            multi_tracking.updateAssignedTracks(tracks, bboxes, centroids, assignments)
            multi_tracking.updateUnassignedTracks(tracks, unassignedTracks)
            multi_tracking.createNewTracks(tracks, bboxes, centroids, unassignedDetections)

        img = cv.imread(cf.dataset_path + '/in000023.jpg')  # in000023 for traffic and in000471 for highway
        if tracks != list():

            # Display the objects. If an object has not been detected
            # in this frame, display its predicted bounding box.
            if tracks != list():
                for track in tracks:
                    car_colour = CAR_COLOURS[track.id % len(CAR_COLOURS)]
                    # for point in track.positions:
                    #    cv.circle(img, (int(point[0]), int(point[1])), 5, car_colour, 1)
                    cv.polylines(img, [np.int32(track.positions)], False, (0, 0, 0), 1)

                    '''for point in track.predictions:
                        vrtx = np.array([[point[0]-5, point[1]-5], [point[0], point[1]+5], [point[0]+5, point[1]-5]],
                                        np.int32)
                        cv.polylines(img, [vrtx], True, car_colour, 1)
                        # cv.rectangle(img, (int(point[0]) - 2, int(point[1]) - 2),
                        #            (int(point[0]) + 2, int(point[1]) + 2), car_colour, 1)'''
                    cv.polylines(img, [np.int32(track.predictions)], False, car_colour, 1)

        cv.imwrite(cf.results_path + '/all_tracks.jpg', img)
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
    visualize_all_tracks(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
