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
from tools.image_parser import get_image_list
from metrics import precision

# Train the network
def background_estimation(cf):

    if cf.save_results:
        # Enable log file
         sys.stdout = Log(cf.log_file)

    print (' ---> Init test: ' + cf.test_name + ' <---')

    #Start trying to load some images and their groundtruth

    # We know how many more images we want to process: cf.nr_images
    # It would be nice to store the images filenames in an array to work more easily

    # Get a list with input images filenames
    imageList = get_image_list(cf.dataset_path,'in', cf.first_image, cf.image_type, cf.nr_images)

    # Get a list with groung truth images filenames
    gtList = get_image_list(cf.gt_path, 'gt', cf.first_image, cf.gt_image_type, cf.nr_images)

    # Get a list with test results filenames
    testList = get_image_list(cf.results_path, str(cf.test_name+'_'), cf.first_image, cf.result_image_type, cf.nr_images)

    # # Display first image
    # img = cv.imread(imageList[0])
    # cv.namedWindow('input image', cv.WINDOW_NORMAL)
    # cv.imshow('input image', img)
    #
    # # Display first ground truth image
    # test_img = cv.imread(testList[0])
    # cv.namedWindow('test image', cv.WINDOW_NORMAL)
    # cv.imshow('test image', test_img)
    #
    # gt_img = cv.imread(gtList[0])
    # cv.namedWindow('gt image', cv.WINDOW_NORMAL)
    # cv.imshow('gt image', gt_img)
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    prec, rec, f1 = precision.evaluate(cf)
    print("PRECISION: "+str(prec))
    print("RECALL: " + str(rec))
    print("F1-SCORE: " + str(f1))
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