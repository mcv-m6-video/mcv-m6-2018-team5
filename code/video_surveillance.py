import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from config.load_configutation import Configuration
from metrics import segmentation_metrics, optical_flow
from tools.image_parser import get_image_list_highway_dataset, get_image_list_kitti_dataset
from tools.log import setup_logging


# Train the network
def background_estimation(cf):

    logger = logging.getLogger(__name__)

    logger.info(' ---> Init test: ' + cf.test_name + ' <---')

    if cf.dataset_name == 'highway':
        # Get a list with input images filenames
        # imageList = get_image_list_highway_dataset(cf.dataset_path, 'in', cf.first_image, cf.image_type, cf.nr_images)

        # Get a list with groung truth images filenames
        gtList = get_image_list_highway_dataset(cf.gt_path, 'gt', cf.first_image, cf.gt_image_type, cf.nr_images)

        # Get a list with test results filenames
        testList = get_image_list_highway_dataset(cf.results_path, str(cf.test_name + '_'), cf.first_image,
                                                  cf.result_image_type, cf.nr_images)

        if cf.segmentation_metrics:
            prec, rec, f1 = segmentation_metrics.evaluate(testList, gtList)
            logger.info("PRECISION: " + str(prec))
            logger.info("RECALL: " + str(rec))
            logger.info("F1-SCORE: " + str(f1))

        if cf.temporal_metrics:
            TP, T, F1_score = segmentation_metrics.temporal_evaluation(testList, gtList)

            plt.plot(TP, label='True Positives')
            plt.plot(T, label='Foreground pixels')
            plt.xlabel('time')
            plt.legend(loc='upper right', fontsize='medium')
            plt.show(block=False)
            if cf.save_results and cf.save_plots:
                plt.savefig(os.path.join(cf.output_folder, "task_2_1.png"))

            plt.close()

            plt.plot(F1_score, label='F1 Score')
            plt.xlabel('time')
            plt.legend(loc='upper right', fontsize='medium')
            plt.show(block=False)
            if cf.save_results and cf.save_plots:
                plt.savefig(os.path.join(cf.output_folder, "task_2_2.png"))

            plt.close()

        if cf.desynchronization:
            F1_score = segmentation_metrics.desynchronization(testList, gtList, cf.desynchronization_frames)

            for i in range(0, len(cf.desynchronization_frames)):
                plt.plot(F1_score[i], label=str(cf.desynchronization_frames[i]) + ' de-synchronization frames')

            plt.xlabel('time')
            plt.legend(loc='upper right', fontsize='medium')
            plt.show(block=False)

            if cf.save_results and cf.save_plots:
                plt.savefig(os.path.join(cf.output_folder, "task_4.png"))

            plt.close()

    if cf.dataset_name == 'kitti':
        # Get a list with input images filenames
        # imageList = get_image_list_kitti_dataset(cf.dataset_path, cf.image_sequences, cf.image_type)

        # Get a list with groung truth images filenames
        gtList = get_image_list_kitti_dataset(cf.gt_path, cf.image_sequences, cf.image_type)

        # Get a list with test results filenames
        testList = get_image_list_kitti_dataset(cf.results_path, cf.image_sequences, cf.image_type, 'LKflow_')

        if cf.evaluate:
            # Call the method to evaluate the optical flow
            msen, pepn, squared_errors, pixel_errors, valid_pixels = optical_flow.evaluate(testList, gtList)
            logger.info('Mean Squared Error: {}'.format(msen))
            logger.info('Percentage of Erroneous Pixels: {}'.format(pepn))

            # Histogram
            for mse, se, vp, seq_name in zip(msen, squared_errors, valid_pixels, cf.image_sequences):
                plt.hist(np.ravel(se[vp]), bins=200, normed=True, color='grey')
                plt.axvline(mse, c='darkred', linestyle=':', label='Mean Squared Error')
                plt.xlabel('Squared Error (Non-occluded areas)')
                plt.ylabel('Frequency')
                plt.title('Sequence {}'.format(seq_name))
                plt.legend()
                plt.show(block=False)
                plt.savefig(os.path.join(cf.output_folder, "task_3_histogram_{}.png".format(seq_name)))

            # for pe in pixel_errors:
            #     plt.hist(pe, bins=20)
            #     plt.show(block=False)

        if cf.plot_optical_flow:
            for test_image, gt_image in zip(testList, gtList):
                optical_flow.plot_optical_flow(test_image)

    logger.info(' ---> Finish test: ' + cf.test_name + ' <---')


# Main function
def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Video surveillance')
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='Configuration file path')
    parser.add_argument('-t', '--test_name', type=str,
                        default=None, help='Name of the test')

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

    # Week 1
    background_estimation(cf)


# Entry point of the script
if __name__ == "__main__":
    main()
