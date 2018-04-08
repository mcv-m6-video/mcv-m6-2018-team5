#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os

import cv2 as cv
import numpy as np
from skimage import io as skio
from tqdm import tqdm

from tools import background_modeling, foreground_improving, detection, visualization, traffic_parameters
from tools.image_parser import get_image_list_changedetection_dataset
from tools.multi_tracking import MultiTracker
from utils.load_configutation import Configuration
from utils.log import log_context
from utils.mkdirs import mkdirs

EPSILON = 1e-8


# noinspection PyUnboundLocalVariable
def road_statistics(cf):
    with log_context(cf.log_file):
        logger = logging.getLogger(__name__)
        logger.info(' ---> Init test: ' + cf.test_name + ' <---')

        if cf.save_results:
            logger.info('Saving results in {}'.format(cf.results_path))
            mkdirs(cf.results_path)

        image_list = get_image_list_changedetection_dataset(
            cf.dataset_path, cf.input_prefix, cf.first_image, cf.image_type, cf.nr_images
        )

        background_img_list = get_image_list_changedetection_dataset(
            cf.dataset_path, cf.input_prefix, cf.first_back, cf.image_type, cf.nr_back
        )

        visualization.visualizeLanes(cv.imread(background_img_list[0]), cf.lanes,
                                     (background_img_list[0].replace('input', 'results').replace(cf.input_prefix, 'lane')))

        visualization.visualizeROI(cv.imread(background_img_list[0]), cf.roi_speed,
                                   (background_img_list[0].replace('input', 'results').replace(cf.input_prefix, 'roi')))

        mean, variance = background_modeling.multivariative_gaussian_modelling(background_img_list, cf.color_space)

        # Instantiate tracker for multi-object tracking
        kalman_init_params = {
            'init_estimate_error': cf.init_estimate_error,
            'motion_model_noise': cf.motion_model_noise,
            'measurement_noise': cf.measurement_noise
        }
        multi_tracker = MultiTracker(cf.cost_of_non_assignment, cf.invisible_too_long,
                                     cf.min_age_threshold, kalman_init_params)
        lane_count = np.zeros((len(cf.lanes), 1))

        for n, image_path in tqdm(enumerate(image_list)):
            image = cv.imread(image_path)
            foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                image, mean, variance, cf.alpha, cf.rho, cf.color_space)

            foreground = foreground_improving.hole_filling(foreground, cf.four_connectivity)
            foreground = foreground_improving.remove_small_regions(foreground, cf.area_filtering_P)
            foreground = foreground_improving.image_opening(foreground, cf.opening_strel, cf.opening_strel_size)
            foreground = foreground_improving.image_closing(foreground, cf.closing_strel, cf.closing_strel_size)

            image_gray = skio.imread(image_path, as_grey=True)
            bboxes, centroids = detection.detectObjects(image_gray, foreground)

            # Tracking
            multi_tracker.predict_new_locations_of_tracks()
            multi_tracker.detection_to_track_assignment(centroids)
            multi_tracker.update_assigned_tracks(bboxes, centroids)
            multi_tracker.update_unassigned_tracks()
            multi_tracker.delete_lost_tracks()
            multi_tracker.create_new_tracks(bboxes, centroids)

            tracks = multi_tracker.tracks

            if n % cf.update_speed == 0:
                for track in tracks:
                    if isinstance(cf.pixels_meter, list):
                        if track.lane != -1:
                            pix_meter = cf.pixels_meter[track.lane]
                        else:
                            pix_meter = sum(cf.pixels_meter)
                    else:
                        pix_meter = cf.pixels_meter
                    if traffic_parameters.is_inside_speed_roi(track.positions[-1], cf.roi_speed):
                        traffic_parameters.speed(track, pix_meter, cf.frames_second, dt=cf.update_speed)
                        track.speeds.append(track.current_speed)
                    if track.lane == -1:
                        vehicle_lane = traffic_parameters.lane_detection(track.positions[-1], cf.lanes)
                        if vehicle_lane != -1:
                            track.lane = vehicle_lane
                            lane_count[vehicle_lane] += 1

            if cf.save_results:
                image_name = os.path.basename(image_path)
                image_name = os.path.splitext(image_name)[0]
                save_path = os.path.join(cf.results_path, image_name + '.' + cf.result_image_type)
                image = image.astype('uint8')
                visualization.displaySpeedResults(image, tracks, cf.max_speed, lane_count, save_path, cf.roi_speed)
        for n, n_lanes in enumerate(lane_count):
            if n+1 == 1:
                logger.info('A total of {} vehicles have passed through the {}st lane'.format(n_lanes, n+1))
            elif n + 1 == 2:
                logger.info('A total of {} vehicles have passed through the {}nd lane'.format(n_lanes, n + 1))
            elif n + 1 == 3:
                logger.info('A total of {} vehicles have passed through the {}rd lane'.format(n_lanes, n + 1))
            else:
                logger.info('A total of {} vehicles have passed through the {}th lane'.format(n_lanes, n + 1))

        logger.info(' ---> Finish test: ' + cf.test_name + ' <---')


# Main function
def main():

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Road statistics [Team 5]')
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
    road_statistics(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
