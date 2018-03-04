from __future__ import division

import argparse
import logging
import os
import sys
import itertools


import cv2 as cv
import numpy as np

from config.load_configutation import Configuration
from tools.image_parser import get_image_list_changedetection_dataset
from tools.log import setup_logging
from metrics.segmentation_metrics import evaluate_single_image

EPSILON = 1e-8

def evaluate_model(imageList, gtList, model):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for image, gt in zip(imageList, gtList):
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        gtImg = cv.imread(image, cv.IMREAD_GRAYSCALE)
        foreground = model.apply(img)
        rect, foreground = cv.threshold(foreground, 50, 1, cv.THRESH_BINARY)
        tp_im, fp_im, tn_im, fn_im = evaluate_single_image(foreground, gtImg)
        tp += tp_im
        fp += fp_im
        tn += tn_im
        fn += fn_im

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall + EPSILON)

    return precision, recall, f1_score

def mog_background_estimator(imageList, gtList, cf):
    # Paper 'An improved adaptive background mixture model for real-time tracking with shadow detection'
    # by P. KadewTraKuPong and R. Bowden in 2001
    logger = logging.getLogger(__name__)
    logger.info('Running adaptive K Gaussian background estimation')

    # Grid search over numGaus and history parameter space
    logger.info('Finding best history and numGaus parameters for adaptive Gaussian model.')
    hist_range = np.linspace(20, 26, 7, dtype='uint8')
    numGaus_range = np.linspace(3, 5, 3, dtype='uint8')

    num_iterations = len(numGaus_range) * len(hist_range)
    logger.info('Running {} iterations'.format(num_iterations))

    score_grid = np.zeros((len(hist_range), len(numGaus_range)))  # score = F1-score
    precision_grid = np.zeros_like(score_grid)  # for representacion purposes
    recall_grid = np.zeros_like(score_grid)  # for representation purposes
    best_parameters = dict(hist=-1, numGaus=-1)
    max_score = 0
    for i, (hist, numGaus) in enumerate(itertools.product(hist_range, numGaus_range)):
        logger.info('[{} of {}]\thist={:.2f}\tnumGaus={:.2f}'.format(i + 1, num_iterations, hist, numGaus))

        # Indices in parameter grid
        i_idx = np.argwhere(hist_range == hist)
        j_idx = np.argwhere(numGaus_range == numGaus)

        if '3.1' in cv.__version__:
            fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history=hist, nmixtures=numGaus)
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorMOG(history=hist, nmixtures=numGaus)
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        precision, recall, f1_score = evaluate_model(imageList, gtList, fgbg)

        # Store them in the array
        score_grid[i_idx, j_idx] = f1_score
        precision_grid[i_idx, j_idx] = precision
        recall_grid[i_idx, j_idx] = recall

        # Compare and select best parameters according to best score
        if f1_score > max_score:
            max_score = f1_score
            best_parameters = dict(hist=hist, numGaus=numGaus)

    logger.info('Finished grid search')
    logger.info('Best parameters: hist={hist}\tnumGaus={numGaus}'.format(**best_parameters))
    logger.info('Best F1-score: {:.3f}'.format(max_score))

    if cf.save_results:
        if '3.1' in cv.__version__:
            fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history=best_parameters.get('hist'), nmixtures=best_parameters.get('numGaus'))
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorMOG(history=best_parameters.get('hist'), nmixtures=best_parameters.get('numGaus'))
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        for image in imageList:
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
            foreground = fgbg.apply(img)
            image_name = os.path.basename(image)
            image_name = os.path.splitext(image_name)[0]
            cv.imwrite(os.path.join(cf.results_path, 'MOG_' + image_name + '.' + cf.result_image_type), foreground)

def mog2_background_estimator(imageList, gtList, cf):
    # Papers 'Improved adaptive Gausian mixture model for background subtraction' by Z.Zivkovic in 2004 and
    # 'Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction' by Z.Zivkovic in 2006
    logger = logging.getLogger(__name__)
    logger.info('Running adaptive multiple Gaussian background estimation')

    # Grid search over varThreshold and hist parameter space
    logger.info('Finding best history and varThreshold parameters for adaptive Gaussian model.')
    hist_range = np.linspace(0, 50, 25, dtype='uint8')
    varThreshold_range = np.linspace(0, 1, 25)

    num_iterations = len(varThreshold_range) * len(hist_range)
    logger.info('Running {} iterations'.format(num_iterations))

    score_grid = np.zeros((len(hist_range), len(varThreshold_range)))  # score = F1-score
    precision_grid = np.zeros_like(score_grid)  # for representacion purposes
    recall_grid = np.zeros_like(score_grid)  # for representation purposes
    best_parameters = dict(hist=-1, varThreshold=-1)
    max_score = 0
    for i, (hist, varThreshold) in enumerate(itertools.product(hist_range, varThreshold_range)):
        logger.info('[{} of {}]\thist={:.2f}\tvarThreshold={:.2f}'.format(i + 1, num_iterations, hist, varThreshold))

        # Indices in parameter grid
        i_idx = np.argwhere(hist_range == hist)
        j_idx = np.argwhere(varThreshold_range == varThreshold)

        if '3.1' in cv.__version__:
            fgbg = cv.createBackgroundSubtractorMOG2(history=best_parameters.get('hist'), varThreshold=best_parameters.get('varThreshold'))
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorMOG2(history=best_parameters.get('hist'), varThreshold=best_parameters.get('varThreshold'))
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        precision, recall, f1_score = evaluate_model(imageList, gtList, fgbg)

        # Store them in the array
        score_grid[i_idx, j_idx] = f1_score
        precision_grid[i_idx, j_idx] = precision
        recall_grid[i_idx, j_idx] = recall

        # Compare and select best parameters according to best score
        if f1_score > max_score:
            max_score = f1_score
            best_parameters = dict(hist=hist, varThreshold=varThreshold)

    logger.info('Finished grid search')
    logger.info('Best parameters: hist={hist}\tvarThreshold={varThreshold}'.format(**best_parameters))
    logger.info('Best F1-score: {:.3f}'.format(max_score))

    if cf.save_results:
        if '3.1' in cv.__version__:
            fgbg = cv.createBackgroundSubtractorMOG2(history=best_parameters.get('hist'), varThreshold=best_parameters.get('varThreshold'))
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorMOG2(history=best_parameters.get('hist'), varThreshold=best_parameters.get('varThreshold'))
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        for image in imageList:
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
            foreground = fgbg.apply(img)
            image_name = os.path.basename(image)
            image_name = os.path.splitext(image_name)[0]
            cv.imwrite(os.path.join(cf.results_path, 'MOG2_' + image_name + '.' + cf.result_image_type), foreground)

def gmg_background_estimator(imageList, gtList, cf):
    # Paper 'Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation'
    # by Andrew B. Godbehere, Akihiro Matsukawa, Ken Goldberg in 2012
    logger = logging.getLogger(__name__)
    logger.info('Running probabilistic background estimation')

    # Grid search over th and initFr parameter space
    logger.info('Finding best initializationFrames and decisionThreshold parameters for adaptive Gaussian model.')
    initFr_range = np.linspace(0, 50, 50, dtype='uint8')
    th_range = np.linspace(0, 1, 20)

    num_iterations = len(th_range) * len(initFr_range)
    logger.info('Running {} iterations'.format(num_iterations))

    score_grid = np.zeros((len(initFr_range), len(th_range)))  # score = F1-score
    precision_grid = np.zeros_like(score_grid)  # for representacion purposes
    recall_grid = np.zeros_like(score_grid)  # for representation purposes
    best_parameters = dict(initFr=-1, th=-1)
    max_score = 0
    for i, (initFr, th) in enumerate(itertools.product(initFr_range, th_range)):
        logger.info('[{} of {}]\tinitFr={:.2f}\tth={:.2f}'.format(i + 1, num_iterations, initFr, th))

        # Indices in parameter grid
        i_idx = np.argwhere(initFr_range == initFr)
        j_idx = np.argwhere(th_range == th)

        if '3.1' in cv.__version__:
            fgbg = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=best_parameters.get('initFr'),
                                                           decisionThreshold=best_parameters.get('th'))
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorGMG(initializationFrames=best_parameters.get('initFr'),
                                              decisionThreshold=best_parameters.get('th'))
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        precision, recall, f1_score = evaluate_model(imageList, gtList, fgbg)

        # Store them in the array
        score_grid[i_idx, j_idx] = f1_score
        precision_grid[i_idx, j_idx] = precision
        recall_grid[i_idx, j_idx] = recall

        # Compare and select best parameters according to best score
        if f1_score > max_score:
            max_score = f1_score
            best_parameters = dict(initFr=initFr, th=th)

    logger.info('Finished grid search')
    logger.info('Best parameters: initFr={initFr}\tth={th}'.format(**best_parameters))
    logger.info('Best F1-score: {:.3f}'.format(max_score))

    if cf.save_results:
        if '3.1' in cv.__version__:
            fgbg = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=best_parameters.get('initFr'),
                                                           decisionThreshold=best_parameters.get('th'))
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorGMG(initializationFrames=best_parameters.get('initFr'),
                                              decisionThreshold=best_parameters.get('th'))
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        for image in imageList:
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
            foreground = fgbg.apply(img)
            image_name = os.path.basename(image)
            image_name = os.path.splitext(image_name)[0]
            cv.imwrite(os.path.join(cf.results_path, 'MOG2_' + image_name + '.' + cf.result_image_type), foreground)

def lsbp_background_estimator(imageList, gtList, cf):
    # Paper 'Background subtraction using local svd binary pattern' by L. Guo in 2016
    logger = logging.getLogger(__name__)
    logger.info('Running local svd binary pattern background estimation')

    if '3.1' in cv.__version__:
        fgbg = cv.bgsegm.createBackgroundSubtractorLSBP()
    elif '2.4' in cv.__version__:
        fgbg = cv.BackgroundSubtractorLSBP()
    else:
        logger.error('OpenCV version not supported')
        sys.exit()

    for image, gt in imageList:
        foreground = fgbg.apply(image)
        if cf.save_results:
            image_name = os.path.basename(image)
            image_name = os.path.splitext(image_name)[0]
            cv.imwrite(os.path.join(cf.results_path, 'LSBP_' + image_name + '.' + cf.result_image_type), foreground)

def main():
    logger = logging.getLogger(__name__)

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Video surveillance application, Team 5')
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

    # Get a list with input images filenames
    imageList = get_image_list_changedetection_dataset(
        cf.dataset_path, 'in', cf.first_image, cf.image_type, cf.nr_images
    )

    # Get a list with groung truth images filenames
    gtList = get_image_list_changedetection_dataset(
        cf.gt_path, 'gt', cf.first_image, cf.gt_image_type, cf.nr_images
    )

    # Run task
    if cf.modelling_method == 'mog':
        mog_background_estimator(imageList, gtList, cf)
    elif cf.modelling_method == 'mog2':
        mog2_background_estimator(imageList, gtList, cf)
    elif cf.modelling_method == 'gmg':
        gmg_background_estimator(imageList, gtList, cf)
    elif cf.modelling_method == 'lsbp':
        lsbp_background_estimator(imageList, gtList, cf)
    else:
        logger.error('Modeling method not implemented')


# Entry point of the script
if __name__ == "__main__":
    main()