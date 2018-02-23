import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Faster plot

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from config.load_configutation import Configuration
from tools.save_log import Log

# Train the network
def background_estimation(cf):

    if cf.save_results:
        # Enable log file
         sys.stdout = Log(cf.log_file)

    print (' ---> Init test: ' + cf.test_name + ' <---')

    #Start trying to load some images and their groundtruth

    # Display first image
    image_path = os.path.join(cf.dataset_path, cf.first_image_to_process + '.' + cf.image_type)
    img = cv.imread(image_path)
    cv.namedWindow(cf.first_image_to_process, cv.WINDOW_NORMAL)
    cv.imshow(cf.first_image_to_process, img)

    # Display first ground truth image
    gt_image_path = os.path.join(cf.gt_path, cf.first_gt_to_process + '.' + cf.gt_image_type)
    gt_img = cv.imread(gt_image_path)
    cv.namedWindow(cf.first_gt_to_process, cv.WINDOW_NORMAL)
    cv.imshow(cf.first_gt_to_process, gt_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # We know how many more images we want to process: cf.nr_images
    # It would be nice to store the images filenames in an array to work more easily



    print (' ---> Finish test: ' + cf.test_name + ' <---')

# Main function
def main():

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Video surveillance')
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='Configuration file path')
    parser.add_argument('-t', '--test_name', type=str,
                        default=None, help='Name of the test')

    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration'\
                                              'path using -c config/path/name'\
                                              ' in the command line'
    assert arguments.test_name is not None, 'Please provide a name for the '\
                                           'test using -e test_name in the '\
                                           'command line'

    # Load the configuration file
    configuration = Configuration(arguments.config_path, arguments.test_name)
    cf = configuration.load()

    # Week 1
    background_estimation(cf)

# Entry point of the script
if __name__ == "__main__":
    main()