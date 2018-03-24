#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os

import numpy as np

from tools.metrics import optical_flow, segmentation as seg_metrics
from tools import visualization
from tools.image_parser import get_image_list_changedetection_dataset, get_image_list_kitti_dataset
from utils.load_configutation import Configuration

EPSILON = 1e-8


def evaluation_metrics(cf):
    logger = logging.getLogger(__name__)

    logger.info(' ---> Init test: ' + cf.test_name + ' <---')

    if cf.dataset_name == 'highway':

        # Get a list with groung truth images filenames
        gt_list = get_image_list_changedetection_dataset(cf.gt_path, 'gt', cf.first_image, cf.gt_image_type,
                                                         cf.nr_images)

        # Get a list with test results filenames
        test_lit = get_image_list_changedetection_dataset(cf.results_path, str(cf.test_name + '_'), cf.first_image,
                                                          cf.result_image_type, cf.nr_images)
        if cf.segmentation_metrics:
            prec, rec, f1 = seg_metrics.evaluate(test_lit, gt_list)
            logger.info("PRECISION: " + str(prec))
            logger.info("RECALL: " + str(rec))
            logger.info("F1-SCORE: " + str(f1))

        if cf.temporal_metrics:
            tp, t, f1_score = seg_metrics.temporal_evaluation(test_lit, gt_list)

            if cf.save_results and cf.save_plots:
                visualization.plot_true_positives(tp, t, cf.output_folder)
                visualization.plot_F1_score(f1_score, cf.output_folder)
            else:
                visualization.plot_true_positives(tp, t)
                visualization.plot_F1_score(f1_score)

        if cf.desynchronization:
            f1_score = seg_metrics.desynchronization(test_lit, gt_list, cf.desynchronization_frames)

            if cf.save_results and cf.save_plots:
                visualization.plot_desynch_vs_time(f1_score, cf.desynchronization_frames, cf.output_folder)
            else:
                visualization.plot_desynch_vs_time(f1_score, cf.desynchronization_frames)

    if cf.dataset_name == 'kitti':

        # Get a list with input images filenames
        image_list = get_image_list_kitti_dataset(cf.dataset_path, cf.image_sequences, cf.image_type)

        # Get a list with groung truth images filenames
        gt_list = get_image_list_kitti_dataset(cf.gt_path, cf.image_sequences, cf.image_type)

        # Get a list with test results filenames
        test_lit = get_image_list_kitti_dataset(cf.results_path, cf.image_sequences, cf.image_type, 'LKflow_')

        if cf.evaluate:
            # Call the method to evaluate the optical flow
            msen, pepn, squared_errors, pixel_errors, valid_pixels = optical_flow.evaluate(test_lit, gt_list)
            logger.info('Mean Squared Error: {}'.format(msen))
            logger.info('Percentage of Erroneous Pixels: {}'.format(pepn))

            for mse, se, pe, vp, seq_name, original_image in zip(msen, squared_errors, pixel_errors, valid_pixels,
                                                                 cf.image_sequences, image_list):
                # Histogram
                visualization.plot_histogram_msen(mse, np.ravel(se[vp]), seq_name, cf.output_folder)
                # Image
                visualization.plot_msen_image(original_image, se, pe, vp, seq_name, cf.output_folder)

        if cf.plot_optical_flow:
            for image, test_image, seq_name in zip(image_list, test_lit, cf.image_sequences):
                # Quiver plot
                output_path = os.path.join(cf.output_folder, 'optical_flow_{}.png'.format(seq_name))
                visualization.plot_optical_flow(image, test_image, cf.optical_flow_downsample, seq_name, output_path)

                # HSV plot
                output_path = os.path.join(cf.output_folder, 'optical_flow_hsv_{}.png'.format(seq_name))
                visualization.plot_optical_flow_hsv(image, test_image, seq_name, output_path)

    logger.info(' ---> Finish test: ' + cf.test_name + ' <---')


def main():

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='W1 - Evaluation of background modeling and optical flow metrics '
                                                 '[Team 5]')
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
    evaluation_metrics(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
