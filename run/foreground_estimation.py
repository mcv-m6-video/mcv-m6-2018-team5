#! /usr/bin/env python

from __future__ import division

import argparse
import logging
import os
import pickle

import cv2 as cv
import numpy as np

from tools.metrics import segmentation as seg_metrics
from tools import background_modeling
from tools import visualization, foreground_improving
from tools.image_parser import get_image_list_changedetection_dataset
from utils.load_configutation import Configuration
from utils.log import log_context
from utils.mkdirs import mkdirs

EPSILON = 1e-8


def foreground_estimation(cf):

    if cf.AUC_area_filtering:
        """ TASK 2 """

        with log_context(cf.log_file):

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

            auc_pr, pixels_range, best_pixels, best_alpha = foreground_improving.area_filtering_auc_vs_pixels(
                cf, background_img_list, foreground_img_list, foreground_gt_list
            )

            visualization.aux_plot_auc_vs_pixels(auc_pr, pixels_range, cf.output_folder)

            # Save auc_pr as a pickle
            auc_pr_path = os.path.join(cf.output_folder, '{}_AUC_vs_pixels.pkl'.format(cf.dataset_name))
            with open(auc_pr_path, 'w') as fd:
                pickle.dump(auc_pr, fd)

            if cf.save_results:
                mkdirs(cf.results_path)
                mean, variance = background_modeling.multivariative_gaussian_modelling(background_img_list,
                                                                                       cf.color_space)
                for (image, gt) in zip(foreground_img_list, foreground_gt_list):
                    foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                        image, mean, variance, best_alpha, cf.rho, cf.color_space
                    )
                    foreground = foreground_improving.hole_filling(foreground, cf.four_connectivity)
                    foreground = foreground_improving.remove_small_regions(foreground, best_pixels)
                    fore = np.array(foreground, dtype='uint8') * 255
                    image_name = os.path.basename(image)
                    image_name = os.path.splitext(image_name)[0]
                    cv.imwrite(
                        os.path.join(cf.results_path, 'task2_' + image_name + '.' + cf.result_image_type),
                        fore)

    else:
        with log_context(cf.log_file):
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

            # Task 1
            mean_back, variance_back = background_modeling.multivariative_gaussian_modelling(background_img_list,
                                                                                   cf.color_space)

            alpha_range = np.linspace(cf.evaluate_alpha_range[0], cf.evaluate_alpha_range[1], num=cf.evaluate_alpha_values)

            precision = []
            recall = []
            F1_score = []
            i = 1
            for alpha in alpha_range:

                tp = 0
                fp = 0
                tn = 0
                fn = 0
                mean = np.copy(mean_back)
                variance = np.copy(variance_back)

                for (image, gt) in zip(foreground_img_list, foreground_gt_list):
                    gt_img = cv.imread(gt, cv.IMREAD_GRAYSCALE)
                    foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                        image, mean, variance, alpha, cf.rho, cf.color_space
                    )

                    foreground = foreground_improving.hole_filling(foreground, cf.four_connectivity)

                    if cf.area_filtering:
                        # Area Filtering
                        foreground = foreground_improving.remove_small_regions(foreground, cf.area_filtering_P)

                    if cf.task_name == 'task3':
                        if cf.dataset_name == 'traffic':
                            foreground = foreground_improving.image_opening(foreground, cf.opening_strel,
                                                                            cf.opening_strel_size)
                            foreground = foreground_improving.image_closing(foreground, cf.closing_strel,
                                                                            cf.closing_strel_size)
                        elif cf.dataset_name == 'fall':
                            foreground = foreground_improving.image_opening(foreground, cf.opening_strel,
                                                                            cf.opening_strel_size)

                            foreground = foreground_improving.image_closing(foreground, cf.closing_strel,
                                                                            cf.closing_strel_size)

                        elif cf.dataset_name == 'highway':
                            foreground = foreground_improving.image_opening(foreground, cf.opening_strel,
                                                                            cf.opening_strel_size)
                            foreground = foreground_improving.image_closing(foreground, cf.closing_strel,
                                                                            cf.closing_strel_size)

                    if cf.shadow_remove:
                        shadow, highlight = foreground_improving.shadow_detection(cf, mean, image, foreground)
                        foreground = foreground - shadow - highlight

                    foreground = np.array(foreground, dtype='uint8')
                    tp_temp, fp_temp, tn_temp, fn_temp = seg_metrics.evaluate_single_image(foreground,
                                                                                           gt_img)

                    tp += tp_temp
                    fp += fp_temp
                    tn += tn_temp
                    fn += fn_temp

                pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * pre * rec / (pre + rec + EPSILON)

                precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                F1_score.append(f1)
                logger.info('[{} of {}]\talpha: {}. F1-score {} '.format(i, len(alpha_range), alpha, f1))
                i += 1

            best_f1_score = max(F1_score)
            index_alpha = F1_score.index(best_f1_score)
            best_alpha = alpha_range[index_alpha]
            logger.info('Best alpha: {:.3f}'.format(best_alpha))
            logger.info('Best F1-score: {:.3f}'.format(best_f1_score))
            visualization.plot_metrics_vs_threshold(precision, recall, F1_score, alpha_range,
                                                    cf.output_folder)

            colors = {
                'highway': 'blue',
                'fall': 'green',
                'traffic': 'orange',
            }
            color = colors.get(cf.dataset_name, 'blue')
            auc_pr = visualization.plot_precision_recall_curve(precision, recall, cf.output_folder, color=color)

            logger.info('Best alpha: {:.3f}'.format(best_alpha))
            logger.info('Best F1-score: {:.3f}'.format(best_f1_score))
            logger.info('AUC: {:.3f}'.format(auc_pr))
            if cf.save_results:
                mean = np.copy(mean_back)
                variance = np.copy(variance_back)
                logger.info('Saving results in {}'.format(cf.results_path))
                mkdirs(cf.results_path)
                for image in foreground_img_list:
                    foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                        image, mean, variance, best_alpha, cf.rho, cf.color_space
                    )
                    foreground = foreground_improving.hole_filling(foreground, cf.four_connectivity)

                    if cf.area_filtering:
                        # Area Filtering
                        foreground = foreground_improving.remove_small_regions(foreground, cf.area_filtering_P)

                    if cf.task_name == 'task3':
                        if cf.dataset_name == 'traffic':
                            foreground = foreground_improving.image_opening(foreground, cf.opening_strel, cf.opening_strel_size)

                            foreground = foreground_improving.image_closing(foreground, cf.closing_strel, cf.closing_strel_size)
                        elif cf.dataset_name == 'fall':
                            foreground = foreground_improving.image_opening(foreground, cf.opening_strel, cf.opening_strel_size)

                            foreground = foreground_improving.image_closing(foreground, cf.closing_strel, cf.closing_strel_size)

                        elif cf.dataset_name == 'highway':
                            foreground = foreground_improving.image_opening(foreground, cf.opening_strel,
                                                                            cf.opening_strel_size)
                            foreground = foreground_improving.image_closing(foreground, cf.closing_strel,
                                                                            cf.closing_strel_size)

                    image_name = os.path.basename(image)
                    image_name = os.path.splitext(image_name)[0]
                    if cf.shadow_remove:
                        shadow, highlight = foreground_improving.shadow_detection(cf, mean, image, foreground)
                        foreground = foreground - shadow - highlight
                        shadow_comp = np.stack([foreground, shadow, highlight], axis=2)
                        cv.imwrite(os.path.join(cf.results_path, 'rgb_' + image_name + '.' + cf.result_image_type),
                                   shadow_comp * 255)
                        cv.imwrite(os.path.join(cf.results_path, 'shadow_' + image_name + '.' + cf.result_image_type),
                               foreground*255)

                    fore = np.array(foreground, dtype='uint8') * 255
                    cv.imwrite(os.path.join(cf.results_path, image_name + '.' + cf.result_image_type), fore)


def main():

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='W3 - Improvement of foreground estimation from background models '
                                                 'in video sequences [Team 5]')
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
    foreground_estimation(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
