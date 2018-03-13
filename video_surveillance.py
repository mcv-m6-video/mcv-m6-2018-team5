#! /usr/bin/env python

from __future__ import division

import argparse
import itertools
import logging
import os
import pickle

import cv2 as cv
import numpy as np

from config.load_configutation import Configuration
from metrics import segmentation_metrics as seg_metrics, optical_flow
from tools import background_modeling
from tools import visualization, foreground_improving
from tools.image_parser import get_image_list_changedetection_dataset, get_image_list_kitti_dataset
from tools.log import log_context
from tools.mkdirs import mkdirs

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
        #setup_logging(cf.log_path)
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
                        foreground = foreground_improving.image_opening(foreground, cf.opening_strel, cf.opening_strel_size)

                        foreground = foreground_improving.image_closing(foreground, cf.closing_strel, cf.closing_strel_size)

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
                        elif  cf.dataset_name == 'fall':
                            pass

                        elif cf.dataset_name == 'highway':
                            pass


                    fore = np.array(foreground, dtype='uint8') * 255
                    image_name = os.path.basename(image)
                    image_name = os.path.splitext(image_name)[0]
                    cv.imwrite(os.path.join(cf.results_path, image_name + '.' + cf.result_image_type), fore)


# Main function
def main():
    # Task choices
    tasks = {
        'evaluate_metrics': evaluation_metrics,
        'background_estimation': background_estimation,
        'foreground_estimation': foreground_estimation,
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
