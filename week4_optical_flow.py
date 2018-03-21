#! /usr/bin/env python

from __future__ import division

import argparse
import itertools
import logging
import os
import pickle
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from config.load_configutation import Configuration
from metrics import optical_flow as of_metrics
from tools import optical_flow as of
from tools import visualization
from tools.image_parser import get_sequence_list_kitti_dataset, get_gt_list_kitti_dataset, \
    get_image_list_changedetection_dataset, get_image_list_ski_video_dataset
from tools.log import log_context
from tools.mkdirs import mkdirs

EPSILON = 1e-8


# noinspection PyUnboundLocalVariable
def optical_flow(cf):
    with log_context(cf.log_file):
        logger = logging.getLogger(__name__)
        logger.info(' ---> Init test: ' + cf.test_name + ' <---')

        if cf.dataset_name == 'kitti':

            # Get a list with input images filenames
            image_list = get_sequence_list_kitti_dataset(cf.dataset_path, cf.image_sequence, cf.image_type)

            # Get a list with ground truth images filenames
            gt_list = get_gt_list_kitti_dataset(cf.gt_path, cf.image_sequence, cf.image_type)

            current_image = image_list[1]
            previous_image = image_list[0]

            if cf.compensation == 'backward':
                reference_image = current_image
                search_image = previous_image
            else:
                reference_image = previous_image
                search_image = current_image

            ref_img_data = cv.imread(reference_image, cv.IMREAD_GRAYSCALE)
            search_img_data = cv.imread(search_image, cv.IMREAD_GRAYSCALE)

            if cf.optimize_block_matching:
                # Placeholder to store results
                optimization_results = dict()
                # GT flow
                optical_flow_gt = cv.imread(gt_list[0], cv.IMREAD_UNCHANGED)

                # Grid search
                total_iters = len(cf.block_size_range) * len(cf.search_area_range)
                for i, (block_size, area_search) in enumerate(
                        itertools.product(cf.block_size_range, cf.search_area_range)
                ):
                    logger.info('[{} / {}] block_size={}\t area_search={}'.format(
                        i + 1, total_iters, block_size, area_search
                    ))

                    _, _, dense_optical_flow, total_time = of.exhaustive_search_block_matching(
                        ref_img_data, search_img_data, block_size, area_search, cf.dfd_norm_type,
                    )
                    msen, pepn, squared_errors, pixel_errors, valid_pixels = of_metrics.flow_errors_MSEN_PEPN(
                        dense_optical_flow, optical_flow_gt
                    )

                    optimization_results[i] = {
                        'block_size': block_size,
                        'area_search': area_search,
                        'msen': msen,
                        'pepn': pepn,
                        'execution_time': total_time,
                    }

                    # Save results every 10 iterations, in case the experiment stops for unknown reasons
                    if i % 10 == 0:
                        output_path = os.path.join(cf.output_folder, '{}_ebma_optimization.pkl'.format(cf.dataset_name))
                        with open(output_path, 'w') as fd:
                            pickle.dump(optimization_results, fd)

            elif cf.sota_opt_flow:
                if cf.sota_opt_flow_option == 'opencv':
                    dense_optical_flow = of.opencv_optflow(
                        ref_img_data, search_img_data, cf.block_size)
                    if dense_optical_flow is None:
                        logger.info('OpenCV version not supported')
                        sys.exit()
                    # Evaluate the optical flow
                    if cf.evaluate:
                        optical_flow_gt = cv.imread(gt_list[0], cv.IMREAD_UNCHANGED)

                        msen, pepn, squared_errors, pixel_errors, valid_pixels = of_metrics.flow_errors_MSEN_PEPN(
                            dense_optical_flow, optical_flow_gt
                        )
                        logger.info('Mean Squared Error: {}'.format(msen))
                        logger.info('Percentage of Erroneous Pixels: {}'.format(pepn))

                        # Histogram
                        visualization.plot_histogram_msen(msen, np.ravel(squared_errors[valid_pixels]),
                                                          cf.image_sequence,
                                                          cf.output_folder)
                        # Image
                        visualization.plot_msen_image(image_list[1], squared_errors, pixel_errors, valid_pixels,
                                                      cf.image_sequence, cf.output_folder)

                    if cf.plot_optical_flow:
                        # Quiver plot
                        output_path = os.path.join(cf.output_folder, 'optical_flow_opencv_{}.png'.format(cf.image_sequence))
                        im = cv.imread(image_list[1], cv.IMREAD_GRAYSCALE)
                        visualization.plot_optical_flow(im, dense_optical_flow, cf.optical_flow_downsample,
                                                        cf.image_sequence, output_path, is_ndarray=True)

                        # HSV plot
                        output_path = os.path.join(cf.output_folder,
                                                   'optical_flow_hsv_{}.png'.format(cf.image_sequence))
                        visualization.plot_optical_flow_hsv(im, dense_optical_flow, cf.image_sequence, output_path,
                                                            is_ndarray=True)

                        output_path = os.path.join(cf.output_folder,
                                                   'optical_flow_middlebury_opencv_{}.png'.format(cf.image_sequence))
                        middlebury = visualization.flow_to_image(dense_optical_flow)
                        plt.figure(figsize=(10, 5), dpi=200)
                        plt.imshow(middlebury)
                        plt.axis('off')
                        plt.show(block=False)
                        plt.savefig(output_path)
                        plt.close()
                    logger.info(' ---> Finish test: ' + cf.test_name + ' <---')

                elif cf.sota_opt_flow_option == 'flownet2':
                    # Evaluate the optical flow
                    if cf.evaluate:
                        # Groun-truth optical flow
                        optical_flow_gt = cv.imread(gt_list[0], cv.IMREAD_UNCHANGED)

                        # Reference image
                        ref_img = cv.imread(image_list[0], cv.IMREAD_GRAYSCALE)

                        # Load pre-computed optical flows with different architecture and training procedures
                        pre_compute_of_folder = cf.output_folder

                        # FlowNet variants
                        flownet_variants = ('css', 's', 'S', 'SD')
                        for fnet_variant in flownet_variants:
                            logger.info('Evaluating FlowNet2-{}'.format(fnet_variant))
                            flow_path = os.path.join(pre_compute_of_folder,
                                                     'flownet2_{}_{}_10.flo'.format(fnet_variant, cf.image_sequence))
                            optical_flow_data = of.read_flow(flow_path)

                            msen, pepn, squared_errors, pixel_errors, valid_pixels = of_metrics.flow_errors_MSEN_PEPN(
                                optical_flow_data, optical_flow_gt
                            )
                            logger.info('Mean Squared Error: {}'.format(msen))
                            logger.info('Percentage of Erroneous Pixels: {}'.format(pepn))

                            # Histogram
                            savefig_path = os.path.join(pre_compute_of_folder, 'flownet2_{}_msen_hist_{}.png'.format(
                                fnet_variant, cf.image_sequence
                            ))
                            visualization.plot_histogram_msen(
                                msen, np.ravel(squared_errors[valid_pixels]), cf.image_sequence, savefig_path
                            )
                            # Image
                            savefig_path = os.path.join(pre_compute_of_folder, 'flownet2_{}_msen_im_{}.png'.format(
                                fnet_variant, cf.image_sequence
                            ))
                            visualization.plot_msen_image(
                                image_list[0], squared_errors, pixel_errors, valid_pixels, cf.image_sequence,
                                savefig_path
                            )

                            if cf.plot_optical_flow:
                                # Quiver plot
                                output_path = os.path.join(
                                    cf.output_folder, 'flownet_{}_optical_flow_{}.png'.format(
                                        fnet_variant, cf.image_sequence
                                    )
                                )
                                visualization.plot_optical_flow(
                                    ref_img, optical_flow_data, cf.optical_flow_downsample, cf.image_sequence,
                                    output_path, is_ndarray=True
                                )

                                # HSV plot
                                output_path = os.path.join(
                                    cf.output_folder, 'flownet_{}_optical_flow_hsv_{}.png'.format(
                                        fnet_variant, cf.image_sequence
                                    )
                                )
                                visualization.plot_optical_flow_hsv(
                                    ref_img, optical_flow_data, cf.image_sequence, output_path, is_ndarray=True
                                )

                                output_path = os.path.join(cf.output_folder,
                                                           'flownet_{}_optical_flow_middlebury_{}.png'.format(
                                        fnet_variant, cf.image_sequence))
                                middlebury = visualization.flow_to_image(optical_flow_data)
                                plt.figure(figsize=(10, 5), dpi=200)
                                plt.imshow(middlebury)
                                plt.axis('off')
                                plt.show(block=False)
                                plt.savefig(output_path)
                                plt.close()


                else:
                    raise ValueError('cv.sota_opt_flow_option {!r} not supported'.format(cf.sota_opt_flow_option))

            else:
                # Run Task 1: Block Matching Algorithm
                predicted_image, optical_flow, dense_optical_flow, _ = of.exhaustive_search_block_matching(
                    ref_img_data, search_img_data, cf.block_size, cf.search_area, cf.dfd_norm_type, verbose=False
                )

                # Evaluate the optical flow
                if cf.evaluate:
                    optical_flow_gt = cv.imread(gt_list[0], cv.IMREAD_UNCHANGED)

                    msen, pepn, squared_errors, pixel_errors, valid_pixels = of_metrics.flow_errors_MSEN_PEPN(
                        dense_optical_flow, optical_flow_gt
                    )
                    logger.info('Mean Squared Error: {}'.format(msen))
                    logger.info('Percentage of Erroneous Pixels: {}'.format(pepn))

                    if cf.plot_prediction:
                        output_path = os.path.join(cf.output_folder, 'optical_flow_prediction_{}.png'.format(
                            cf.image_sequence
                        ))
                        plt.imshow(predicted_image, cmap='gray')
                        plt.show(block=False)
                        plt.savefig(output_path)
                        plt.close()

                    # Histogram
                    visualization.plot_histogram_msen(msen, np.ravel(squared_errors[valid_pixels]), cf.image_sequence,
                                                      cf.output_folder)
                    # Image
                    visualization.plot_msen_image(image_list[1], squared_errors, pixel_errors, valid_pixels,
                                                  cf.image_sequence, cf.output_folder)



                if cf.plot_optical_flow:
                    # Quiver plot
                    output_path = os.path.join(cf.output_folder, 'optical_flow_{}.png'.format(cf.image_sequence))
                    im = cv.imread(image_list[1], cv.IMREAD_GRAYSCALE)
                    visualization.plot_optical_flow(im, dense_optical_flow, cf.optical_flow_downsample,
                                                    cf.image_sequence, output_path, is_ndarray=True)

                    # HSV plot
                    output_path = os.path.join(cf.output_folder, 'optical_flow_hsv_{}.png'.format(cf.image_sequence))
                    visualization.plot_optical_flow_hsv(im, dense_optical_flow, cf.image_sequence, output_path,
                                                        is_ndarray=True)

        elif cf.dataset_name == 'traffic':

            image_list = get_image_list_changedetection_dataset(cf.dataset_path, 'in', cf.first_image, cf.image_type,
                                                                cf.nr_images)

            gt_list = get_image_list_changedetection_dataset(cf.gt_path, 'gt', cf.first_image, cf.gt_image_type,
                                                             cf.nr_images)

            if cf.save_results:
                logger.info('Saving results in {}'.format(cf.results_path))
                mkdirs(cf.results_path)

            if cf.sota_video_stab:
                of.video_stabilization_sota2(image_list, cf.output_folder)
                '''prev_to_cur_transform = []
                previous_image = image_list[0]
                prev_image = cv.imread(previous_image, cv.IMREAD_GRAYSCALE)
                prev_corner = cv.goodFeaturesToTrack(prev_image, maxCorners=200, qualityLevel=0.01, minDistance=30.0,
                                                     blockSize=3)
                for idx in range(1, len(image_list)):
                    current_image = image_list[idx]
                    previous_image = image_list[idx - 1]
                    curr_image = cv.imread(current_image, cv.IMREAD_GRAYSCALE)
                    prev_image = cv.imread(previous_image, cv.IMREAD_GRAYSCALE)
                    prev_to_cur_transform = of.video_stabilization_sota(prev_image, curr_image,
                                                                                                 prev_to_cur_transform,
                                                                                                 prev_corner)
                # convert list of transforms to array
                prev_to_cur_transform = np.array(prev_to_cur_transform)
                # cumsum of all transforms for trajectory
                trajectory = np.cumsum(prev_to_cur_transform, axis=0)

                # convert trajectory array to df
                trajectory = pd.DataFrame(trajectory)
                # rolling mean to smooth
                smoothed_trajectory = trajectory.rolling(window=30, center=False).mean()
                # back fill nas caused by smoothing
                smoothed_trajectory = smoothed_trajectory.fillna(method='bfill')
                # new set of prev to cur transform, removing trajectory and replacing w/smoothed
                new_prev_to_cur_transform = prev_to_cur_transform + (smoothed_trajectory - trajectory)

                # initialize transformation matrix
                T = np.zeros((2, 3))
                # convert transform df to array
                new_prev_to_cur_transform = np.array(new_prev_to_cur_transform)

                for k in range(len(image_list) - 1):
                    cur = cv.imread(image_list[k])
                    T[0, 0] = np.cos(new_prev_to_cur_transform[k][2])
                    T[0, 1] = -np.sin(new_prev_to_cur_transform[k][2])
                    T[1, 0] = np.sin(new_prev_to_cur_transform[k][2])
                    T[1, 1] = np.cos(new_prev_to_cur_transform[k][2])
                    T[0, 2] = new_prev_to_cur_transform[k][0]
                    T[1, 2] = new_prev_to_cur_transform[k][1]
                    # apply saved transform (resource: http://nghiaho.com/?p=2208)
                    rect_image = cv.warpAffine(cur, T, (cur.shape[0], cur.shape[1]))
                    if cf.save_results:
                        image_name = os.path.basename(image_list[k])
                        image_name = os.path.splitext(image_name)[0]
                        cv.imwrite(os.path.join(cf.results_path, image_name + '.' + cf.result_image_type), rect_image)'''

            else:
                acc_u = 0
                acc_v = 0
                # Running average of directions
                previous_u = 0
                previous_v = 0

                if cf.save_results:
                    # Create folder to store histograms
                    histogram_folder = os.path.join(cf.results_path, 'histograms')
                    mkdirs(histogram_folder)

                for idx in range(1, len(image_list)):
                    current_image = image_list[idx]
                    previous_image = image_list[idx - 1]

                    if cf.compensation == 'backward':
                        reference_image = current_image
                        search_image = previous_image
                    else:
                        reference_image = previous_image
                        search_image = current_image

                    ref_img_data = cv.imread(reference_image, cv.IMREAD_GRAYSCALE)
                    search_img_data = cv.imread(search_image, cv.IMREAD_GRAYSCALE)

                    save_path = os.path.join(cf.output_folder, '{}_{}_{}_{}.pkl'.format(
                        idx, cf.block_size, cf.search_area, cf.compensation
                    ))
                    if cf.sota_opt_flow and cf.sota_opt_flow_option == 'opencv':
                        dense_flow = of.opencv_optflow(
                            ref_img_data, search_img_data, cf.block_size)
                    else:
                        try:
                            with open(save_path, 'rb') as file_flow:
                                dense_flow = pickle.load(file_flow)
                        except Exception:
                            with open(save_path, 'wb') as fd:
                                _, opt_flow, dense_flow, _ = of.exhaustive_search_block_matching(
                                    ref_img_data, search_img_data, cf.block_size, cf.search_area, cf.dfd_norm_type,
                                    verbose=False)
                                pickle.dump(dense_flow, fd)

                    image_data = cv.imread(gt_list[idx])

                    # Params
                    strategy = 'max'  # 'max', 'trimmed_mean', 'background_block'
                    if strategy == 'background_blocks':
                        center_positions = [(30, 290), (210, 30)]
                        neighborhood = 20
                        additional_params = {
                            'center_positions': center_positions,
                            'neighborhood': neighborhood,
                        }
                    else:
                        additional_params = dict()
                    running_avg = 0

                    # Run
                    rect_image, acc_direction, previous_direction = of.video_stabilization(
                        image_data, dense_flow, cf.compensation, strategy, cf.search_area,
                        (acc_u, acc_v), (previous_u, previous_v), running_avg, **additional_params
                    )
                    acc_u, acc_v = acc_direction
                    previous_u, previous_v = previous_direction

                    if cf.save_results:
                        image_name = os.path.basename(current_image)
                        image_name = os.path.splitext(image_name)[0]
                        cv.imwrite(os.path.join(cf.results_path, image_name + '.' + cf.result_image_type), rect_image)
                        if cf.save_plots:
                            if strategy == 'background_blocks':
                                fig = plt.figure(figsize=(10, 10))
                                ax = fig.add_subplot(111)
                                plt.imshow(image_data)
                                for center_position, color in zip(center_positions, itertools.cycle(['r', 'y', 'b'])):
                                    ax.scatter(center_position[1], center_position[0], s=8, marker='x', color=color,
                                               label='Pixel position ({0}, {1})'.format(*center_position))
                                    rectangle_center = (center_position[1] - neighborhood,
                                                        center_position[0] - neighborhood)
                                    ax.add_patch(
                                        patches.Rectangle(
                                            rectangle_center,
                                            neighborhood * 2,
                                            neighborhood * 2,
                                            fill=False, color=color
                                        )
                                    )
                                plt.legend()
                                plt.axis('off')
                                plot_path = os.path.join(
                                    histogram_folder,
                                    image_name + '_block_markers.' + cf.result_image_type
                                )
                                plt.savefig(plot_path)
                                plt.close()

                            # Save histogram of directions
                            hist_path = os.path.join(histogram_folder, image_name + '_hist_2d.' + cf.result_image_type)
                            visualization.plot_optical_flow_histogram(dense_flow, cf.search_area, hist_path)

            logger.info(' ---> Finish test: ' + cf.test_name + ' <---')

        elif cf.dataset_name == 'ski_video':

            image_list = get_image_list_ski_video_dataset(cf.dataset_path, cf.first_image, cf.image_type, cf.nr_images)

            if cf.save_results:
                logger.info('Saving results in {}'.format(cf.results_path))
                mkdirs(cf.results_path)

            acc_u = 0
            acc_v = 0
            # Running average of directions
            previous_u = 0
            previous_v = 0

            if cf.save_results:
                # Create folder to store histograms
                histogram_folder = os.path.join(cf.results_path, 'histograms')
                mkdirs(histogram_folder)

            for idx in range(1, len(image_list)):
                # Reset accumulator every 5 frames
                if idx % 5 == 0:
                    acc_u = 0
                    acc_v = 0

                current_image = image_list[idx]
                previous_image = image_list[idx - 1]

                if cf.compensation == 'backward':
                    reference_image = current_image
                    search_image = previous_image
                else:
                    reference_image = previous_image
                    search_image = current_image

                ref_img_data = cv.imread(reference_image, cv.IMREAD_GRAYSCALE)
                search_img_data = cv.imread(search_image, cv.IMREAD_GRAYSCALE)

                save_path = os.path.join(cf.output_folder, '{}_{}_{}_{}.pkl'.format(
                    idx, cf.block_size, cf.search_area, cf.compensation
                ))

                try:
                    with open(save_path, 'rb') as file_flow:
                        dense_flow = pickle.load(file_flow)
                except Exception:
                    with open(save_path, 'wb') as fd:
                        _, opt_flow, dense_flow, _ = of.exhaustive_search_block_matching(
                            ref_img_data, search_img_data, cf.block_size, cf.search_area, cf.dfd_norm_type,
                            verbose=False)
                        pickle.dump(dense_flow, fd)

                image_data = cv.imread(current_image, cv.IMREAD_COLOR)

                # Params
                strategy = 'max'  # 'max', 'trimmed_mean', 'background_block'
                if strategy == 'background_blocks':
                    center_positions = [(30, 290), (210, 30)]
                    neighborhood = 20
                    additional_params = {
                        'center_positions': center_positions,
                        'neighborhood': neighborhood,
                    }
                else:
                    additional_params = dict()
                running_avg = 0

                # Run
                rect_image, acc_direction, previous_direction = of.video_stabilization(
                    image_data, dense_flow, cf.compensation, strategy, cf.search_area,
                    (acc_u, acc_v), (previous_u, previous_v), running_avg, **additional_params
                )
                acc_u, acc_v = acc_direction
                previous_u, previous_v = previous_direction

                if cf.save_results:
                    image_name = os.path.basename(current_image)
                    image_name = os.path.splitext(image_name)[0]
                    cv.imwrite(os.path.join(cf.results_path, image_name + '.' + cf.result_image_type), rect_image)
                    if cf.save_plots:
                        if strategy == 'background_blocks':
                            fig = plt.figure(figsize=(10, 10))
                            ax = fig.add_subplot(111)
                            plt.imshow(image_data)
                            for center_position, color in zip(center_positions, itertools.cycle(['r', 'y', 'b'])):
                                ax.scatter(center_position[1], center_position[0], s=8, marker='x', color=color,
                                           label='Pixel position ({0}, {1})'.format(*center_position))
                                rectangle_center = (center_position[1] - neighborhood,
                                                    center_position[0] - neighborhood)
                                ax.add_patch(
                                    patches.Rectangle(
                                        rectangle_center,
                                        neighborhood * 2,
                                        neighborhood * 2,
                                        fill=False, color=color
                                    )
                                )
                            plt.legend()
                            plt.axis('off')
                            plot_path = os.path.join(
                                histogram_folder,
                                image_name + '_block_markers.' + cf.result_image_type
                            )
                            plt.savefig(plot_path)
                            plt.close()

                        # Save histogram of directions
                        hist_path = os.path.join(histogram_folder, image_name + '_hist_2d.' + cf.result_image_type)
                        visualization.plot_optical_flow_histogram(dense_flow, cf.search_area, hist_path)

            logger.info(' ---> Finish test: ' + cf.test_name + ' <---')


# Main function
def main():
    # Task choices
    tasks = {
        'optical_flow': optical_flow
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
