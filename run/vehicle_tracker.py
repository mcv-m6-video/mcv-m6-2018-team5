#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os

from skimage import io as skio
from skimage import measure as skmeasure
from skimage import morphology as skmorph

from tools import background_modeling, foreground_improving, visualization
from tools.image_parser import get_image_list_changedetection_dataset
from tools.tracking import Tracker
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
        tracker = Tracker(cf.distance_threshold, cf.max_frames_to_skip, cf.max_trace_length, 0, cf)

        for image_path in foreground_img_list:

            foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                image_path, mean, variance, cf.alpha, cf.rho, cf.color_space)

            foreground = foreground_improving.hole_filling(foreground, cf.four_connectivity)
            foreground = foreground_improving.remove_small_regions(foreground, cf.area_filtering_P)
            foreground = foreground_improving.image_opening(foreground, cf.opening_strel, cf.opening_strel_size)
            foreground = foreground_improving.image_closing(foreground, cf.closing_strel, cf.closing_strel_size)

            # Detect distinct objects in the image
            detections = list()
            labeled_image = skmorph.label(foreground, connectivity=foreground.ndim)
            img_data = skio.imread(image_path, as_grey=True)
            region_properties = skmeasure.regionprops(labeled_image, img_data)

            # (Optional) Filter regions
            filtered_region_props = region_properties

            # Extract detections from region properties
            # TODO: filter detections properly....
            #detections = [list(x.centroid) for x in filtered_region_props]
            detections = [list(x.centroid) if(x.convex_area > 2000) else list() for x in filtered_region_props]
            idx = 0
            while idx < len(detections):
                if detections[idx] == list():
                    detections.remove(detections[idx])
                else:
                    idx += 1
            # Update tracking
            tracker.update(detections)

            # TODO: add speed estimation


            if cf.save_results:
                image_name = os.path.basename(image_path)
                image_name = os.path.splitext(image_name)[0]

                # TODO: add bounding box drawing with car id and speed
                # (Optional) Plot detections
                save_path = os.path.join(cf.results_path, image_name + '.' + cf.result_image_type)
                visualization.show_detections(img_data, labeled_image, region_properties, save_path)

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
