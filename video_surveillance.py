#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os

import cv2 as cv
import numpy as np

from config.load_configutation import Configuration
from metrics import segmentation_metrics, optical_flow
from tools import background_modeling
from tools import visualization
from tools.image_parser import get_image_list_changedetection_dataset, get_image_list_kitti_dataset
from tools.log import setup_logging


def evaluation_metrics(cf):
    logger = logging.getLogger(__name__)

    logger.info(' ---> Init test: ' + cf.test_name + ' <---')

    if cf.dataset_name == 'highway':

        # Get a list with groung truth images filenames
        gtList = get_image_list_changedetection_dataset(cf.gt_path, 'gt', cf.first_image, cf.gt_image_type,
                                                        cf.nr_images)

        # Get a list with test results filenames
        testList = get_image_list_changedetection_dataset(cf.results_path, str(cf.test_name + '_'), cf.first_image,
                                                          cf.result_image_type, cf.nr_images)
        if cf.segmentation_metrics:
            prec, rec, f1 = segmentation_metrics.evaluate(testList, gtList)
            logger.info("PRECISION: " + str(prec))
            logger.info("RECALL: " + str(rec))
            logger.info("F1-SCORE: " + str(f1))

        if cf.temporal_metrics:
            TP, T, F1_score = segmentation_metrics.temporal_evaluation(testList, gtList)

            if cf.save_results and cf.save_plots:
                visualization.plot_true_positives(TP, T, cf.output_folder)
                visualization.plot_F1_score(F1_score, cf.output_folder)
            else:
                visualization.plot_true_positives(TP, T)
                visualization.plot_F1_score(F1_score)

        if cf.desynchronization:
            F1_score = segmentation_metrics.desynchronization(testList, gtList, cf.desynchronization_frames)

            if cf.save_results and cf.save_plots:
                visualization.plot_desynch_vs_time(F1_score, cf.desynchronization_frames, cf.output_folder)
            else:
                visualization.plot_desynch_vs_time(F1_score, cf.desynchronization_frames)

    if cf.dataset_name == 'kitti':

        # Get a list with input images filenames
        imageList = get_image_list_kitti_dataset(cf.dataset_path, cf.image_sequences, cf.image_type)

        # Get a list with groung truth images filenames
        gtList = get_image_list_kitti_dataset(cf.gt_path, cf.image_sequences, cf.image_type)

        # Get a list with test results filenames
        testList = get_image_list_kitti_dataset(cf.results_path, cf.image_sequences, cf.image_type, 'LKflow_')

        if cf.evaluate:
            # Call the method to evaluate the optical flow
            msen, pepn, squared_errors, pixel_errors, valid_pixels = optical_flow.evaluate(testList, gtList)
            logger.info('Mean Squared Error: {}'.format(msen))
            logger.info('Percentage of Erroneous Pixels: {}'.format(pepn))

            for mse, se, pe, vp, seq_name, original_image in zip(msen, squared_errors, pixel_errors, valid_pixels,
                                                                 cf.image_sequences, imageList):
                # Histogram
                visualization.plot_histogram_msen(mse, np.ravel(se[vp]), seq_name, cf.output_folder)
                # Image
                visualization.plot_msen_image(original_image, se, pe, vp, seq_name, cf.output_folder)

        if cf.plot_optical_flow:
            for image, test_image, seq_name in zip(imageList, testList, cf.image_sequences):
                # Quiver plot
                output_path = os.path.join(cf.output_folder, 'optical_flow_{}.png'.format(seq_name))
                visualization.plot_optical_flow(image, test_image, cf.optical_flow_downsample, seq_name, output_path)

                # HSV plot
                output_path = os.path.join(cf.output_folder, 'optical_flow_hsv_{}.png'.format(seq_name))
                visualization.plot_optical_flow_hsv(image, test_image, seq_name, output_path)

    logger.info(' ---> Finish test: ' + cf.test_name + ' <---')


def background_estimation(cf):
    logger = logging.getLogger(__name__)

    # Get a list with input images filenames
    imageList = get_image_list_changedetection_dataset(cf.dataset_path, 'in', cf.first_image, cf.image_type,
                                                       cf.nr_images)

    """ GAUSSIAN MODELLING """
    alpha = 1
    logger.info('Single gaussian modeling with parameter: alpha={}'.format(alpha))
    mean, variance = background_modeling.single_gaussian_modelling(imageList[:len(imageList) // 2])

    for image in imageList[(len(imageList) // 2):]:
        foreground = background_modeling.foreground_estimation(image, mean, variance, alpha)
        if cf.save_results:
            image = int(cf.first_image)
            fore = np.array(foreground, dtype='uint8')
            cv.imwrite(os.path.join(cf.output_folder, cf.dataset_name, str(image) + '.png'), fore)
            image += 1

    """ ADAPTIVE MODELLING """
    rho = 0.5
    alpha = 1.1
    logger.info('Single Gaussian adaptive modeling with parameters: alpha={}, rho={}'.format(alpha, rho))
    foregrounds = background_modeling.adaptive_foreground_estimation(
        imageList[(len(imageList) // 2):], mean, variance, alpha, rho
    )

    if cf.save_results:
        image = int(cf.first_image)
        for fore in foregrounds:
            fore = np.array(fore, dtype='uint8')
            cv.imwrite(os.path.join(cf.output_folder, cf.dataset_name, 'ADAPTIVE_'+str(image) + '.png'), fore*255)
            image += 1


# Main function
def main():
    # Task choices
    tasks = {
        'evaluate_metrics': evaluation_metrics,
        'background_estimation': background_estimation,
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

    # Set up logging
    if cf.save_results:
        log_file = cf.log_file
    else:
        log_file = None
    setup_logging(log_file)

    # Run task
    task_fn = tasks[arguments.task]
    task_fn(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
