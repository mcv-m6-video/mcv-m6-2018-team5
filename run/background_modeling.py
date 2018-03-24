#! /usr/bin/env python

from __future__ import division

import argparse
import itertools
import logging
import os

import cv2 as cv
import numpy as np

from tools.metrics import segmentation as seg_metrics
from tools import background_modeling
from tools import visualization
from tools.image_parser import get_image_list_changedetection_dataset
from utils.load_configutation import Configuration
from utils.mkdirs import mkdirs

EPSILON = 1e-8


def background_estimation(cf):
    logger = logging.getLogger(__name__)

    # Get a list with input images filenames
    image_list = get_image_list_changedetection_dataset(
        cf.dataset_path, 'in', cf.first_image, cf.image_type, cf.nr_images
    )

    # Get a list with groung truth images filenames
    gt_list = get_image_list_changedetection_dataset(
        cf.gt_path, 'gt', cf.first_image, cf.gt_image_type, cf.nr_images
    )
    background_img_list = image_list[:len(image_list) // 2]
    foreground_img_list = image_list[(len(image_list) // 2):]
    foreground_gt_list = gt_list[(len(image_list) // 2):]

    if cf.plot_back_model:
        output_path = os.path.join(cf.output_folder, 'gaussian_model')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        visualization.plot_back_evolution(background_img_list, cf.first_image, output_path)

    if cf.evaluate_foreground:
        logger.info('Running foreground evaluation')
        if cf.color_images:
            mean, variance = background_modeling.multivariative_gaussian_modelling(background_img_list,
                                                                                   cf.color_space)
        else:
            # Model with a single Gaussian
            mean, variance = background_modeling.single_gaussian_modelling(background_img_list)

        alpha_range = np.linspace(cf.evaluate_alpha_range[0], cf.evaluate_alpha_range[1], num=cf.evaluate_alpha_values)
        precision, recall, f1_score, false_positive_rate = seg_metrics.evaluate_foreground_estimation(
            cf.modelling_method, foreground_img_list, foreground_gt_list, mean, variance, alpha_range, cf.rho,
            cf.color_images, cf.color_space
        )

        best_f1_score = max(f1_score)
        index_alpha = f1_score.index(best_f1_score)
        best_alpha = alpha_range[index_alpha]
        logger.info('Best alpha: {:.3f}'.format(best_alpha))
        logger.info('Best F1-score: {:.3f}'.format(best_f1_score))

        visualization.plot_metrics_vs_threshold(precision, recall, f1_score, alpha_range, cf.output_folder)

        colors = {
            'highway': 'blue',
            'fall': 'green',
            'traffic': 'orange',
        }
        color = colors.get(cf.dataset_name, 'blue')
        auc_pr = visualization.plot_precision_recall_curve(precision, recall, cf.output_folder, color=color)

        logger.info("AUC: {}".format(auc_pr))

        for alpha_value, prec, rec, f1 in zip(alpha_range, precision, recall, f1_score):
            logger.info(
                '[alpha={:.2f}]   precision={}    recall={}    f1={}'.format(
                    alpha_value, prec, rec, f1
                )
            )

    else:
        if cf.color_images:
            mean, variance = background_modeling.multivariative_gaussian_modelling(background_img_list,
                                                                                   cf.color_space)
        else:
            # Model with a single Gaussian
            mean, variance = background_modeling.single_gaussian_modelling(background_img_list)

        if cf.modelling_method == 'non-adaptive':
            logger.info('Running single Gaussian background estimation')
            if cf.save_results:
                logger.info('Saving results in {}'.format(cf.results_path))
                mkdirs(cf.results_path)
                for image in foreground_img_list:
                    if cf.color_images:
                        foreground = background_modeling.foreground_estimation_color(image, mean, variance, cf.alpha,
                                                                                     cf.color_space)
                    else:
                        foreground = background_modeling.foreground_estimation(image, mean, variance, cf.alpha)
                    image_name = os.path.basename(image)
                    image_name = os.path.splitext(image_name)[0]
                    fore = np.array(foreground, dtype='uint8') * 255
                    cv.imwrite(os.path.join(cf.results_path, image_name + '.' + cf.result_image_type), fore)

        elif cf.modelling_method == 'adaptive':
            logger.info('Running adaptive single Gaussian background estimation')

            if cf.find_best_parameters:
                # Grid search over rho and alpha parameter space
                logger.info('Finding best alpha and rho parameters for adaptive Gaussian model.')
                alpha_range = np.linspace(
                    cf.evaluate_alpha_range[0], cf.evaluate_alpha_range[1], cf.evaluate_alpha_values
                )
                rho_range = np.linspace(
                    cf.evaluate_rho_range[0], cf.evaluate_rho_range[1], cf.evaluate_rho_values
                )
                num_iterations = len(rho_range) * len(alpha_range)
                logger.info('Running {} iterations'.format(num_iterations))

                score_grid = np.zeros((len(alpha_range), len(rho_range)))  # score = F1-score
                precision_grid = np.zeros_like(score_grid)  # for representacion purposes
                recall_grid = np.zeros_like(score_grid)  # for representation purposes
                best_parameters = dict(alpha=-1, rho=-1)
                max_score = 0
                for i, (alpha, rho) in enumerate(itertools.product(alpha_range, rho_range)):
                    logger.info('[{} of {}]\talpha={:.2f}\trho={:.2f}'.format(i + 1, num_iterations, alpha, rho))

                    # Indices in parameter grid
                    i_idx = np.argwhere(alpha_range == alpha)
                    j_idx = np.argwhere(rho_range == rho)

                    # Compute evaluation metrics for this combination of parameters
                    _, _, _, _, precision, recall, f1_score, fpr = seg_metrics.evaluate_list_foreground_estimation(
                        cf.modelling_method, foreground_img_list, foreground_gt_list, mean, variance, alpha, rho,
                        cf.color_images, cf.color_space
                    )
                    # Store them in the array
                    score_grid[i_idx, j_idx] = f1_score
                    precision_grid[i_idx, j_idx] = precision
                    recall_grid[i_idx, j_idx] = recall

                    # Compare and select best parameters according to best score
                    if f1_score > max_score:
                        max_score = f1_score
                        best_parameters = dict(alpha=alpha, rho=rho)

                logger.info('Finished grid search')
                logger.info('Best parameters: alpha={alpha}\trho={rho}'.format(**best_parameters))
                logger.info('Best F1-score: {:.3f}'.format(max_score))

                visualization.plot_adaptive_gaussian_grid_search(score_grid, alpha_range, rho_range,
                                                                 best_parameters, best_score=max_score,
                                                                 metric='F1-score', sequence_name=cf.dataset_name)

            if cf.save_results:
                logger.info('Saving results in {}'.format(cf.results_path))
                mkdirs(cf.results_path)
                for image in foreground_img_list:
                    if cf.color_images:
                        foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                            image, mean, variance, cf.alpha, cf.rho, cf.color_space
                        )
                    else:
                        foreground, mean, variance = background_modeling.adaptive_foreground_estimation(
                            image, mean, variance, cf.alpha, cf.rho
                        )
                    image_name = os.path.basename(image)
                    image_name = os.path.splitext(image_name)[0]
                    fore = np.array(foreground, dtype='uint8') * 255
                    cv.imwrite(os.path.join(cf.results_path, 'ADAPTIVE_' + image_name + '.' + cf.result_image_type),
                               fore)


def main():

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='W2 - Techniques to model background and estimate foreground in '
                                                 'video sequences [Team 5]')
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
    background_estimation(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
