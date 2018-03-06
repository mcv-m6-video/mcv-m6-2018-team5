from __future__ import division

import argparse
import logging
import os
import sys
import itertools

from sklearn.metrics import auc
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from config.load_configutation import Configuration
from tools.image_parser import get_image_list_changedetection_dataset
from tools.log import setup_logging
from metrics.segmentation_metrics import evaluate_single_image
from tools.visualization import plot_AUC_curve
from tools import Subsense

EPSILON = 1e-8

def evaluate_model(imageList, gtList, model):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for image, gt in zip(imageList, gtList):
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        gtImg = cv.imread(gt, cv.IMREAD_GRAYSCALE)
        foreground = model.apply(img)
        rect, foreground = cv.threshold(foreground, 50, 1, cv.THRESH_BINARY)
        if np.min(foreground) != np.max(foreground):
            tp_im, fp_im, tn_im, fn_im = evaluate_single_image(foreground, gtImg)
            tp += tp_im
            fp += fp_im
            tn += tn_im
            fn += fn_im


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * tpr / (precision + tpr + EPSILON)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr, f1_score

def mog_background_estimator(imageList, gtList, cf):
    # Paper 'An improved adaptive background mixture model for real-time tracking with shadow detection'
    # by P. KadewTraKuPong and R. Bowden in 2001
    logger = logging.getLogger(__name__)
    logger.info('Running adaptive K Gaussian background estimation')

    # Grid search over varThreshold and hist parameter space
    logger.info('Finding best decisionThreshold parameter.')
    th_range = np.linspace(0, 50, 100)

    num_iterations = len(th_range)
    logger.info('Running {} iterations'.format(num_iterations))

    tpr = []
    fpr = []

    best_th = -1
    max_score = 0
    for i, (th) in enumerate(th_range):
        logger.info('[{} of {}]\tvarThreshold={:.2f}'.format(i + 1, num_iterations, th))

        if '3.1' in cv.__version__:
            fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history=25, backgroundRatio=th)
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorMOG(history=25, backgroundRatio=th)
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        tpr_temp, fpr_temp, f1_score = evaluate_model(imageList, gtList, fgbg)

        # Store them in the array
        tpr.append(tpr_temp)
        fpr.append(fpr_temp)

        # Compare and select best parameters according to best score
        if f1_score > max_score:
            max_score = f1_score
            best_th = th

    area = plot_AUC_curve(tpr, fpr, cf.output_folder)

    logger.info('Finished search')
    logger.info('Best threshold: {:-3f}'.format(best_th))
    logger.info('Best F1-score: {:.3f}'.format(max_score))
    logger.info('AUC: {:.3f}'.format(area))

    if cf.save_results:
        if '3.1' in cv.__version__:
            fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history=25, backgroundRatio=best_th)
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorMOG(history=25, backgroundRatio=best_th)
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
    logger.info('Finding best decisionThreshold parameter.')
    th_range = np.linspace(0, 50, 100)

    num_iterations = len(th_range)
    logger.info('Running {} iterations'.format(num_iterations))

    tpr = []
    fpr = []

    best_th = -1
    max_score = 0
    for i, (th) in enumerate(th_range):
        logger.info('[{} of {}]\tvarThreshold={:.2f}'.format(i + 1, num_iterations, th))

        if '3.1' in cv.__version__:
            fgbg = cv.createBackgroundSubtractorMOG2(history=25, varThreshold=th, detectShadows=False)
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorMOG2(history=25, varThreshold=th, bShadowDetection=False)
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        tpr_temp, fpr_temp, f1_score = evaluate_model(imageList, gtList, fgbg)

        # Store them in the array
        tpr.append(tpr_temp)
        fpr.append(fpr_temp)

        # Compare and select best parameters according to best score
        if f1_score > max_score:
            max_score = f1_score
            best_th = th

    area = plot_AUC_curve(tpr, fpr, cf.output_folder)

    logger.info('Finished search')
    logger.info('Best threshold: {:-3f}'.format(best_th))
    logger.info('Best F1-score: {:.3f}'.format(max_score))
    logger.info('AUC: {:.3f}'.format(area))

    if cf.save_results:
        if '3.1' in cv.__version__:
            fgbg = cv.createBackgroundSubtractorMOG2(history=25, varThreshold=best_th, detectShadows=False)
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorMOG2(history=25, varThreshold=best_th, bShadowDetection=False)
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

    # Grid search over varThreshold and hist parameter space
    logger.info('Finding best decisionThreshold parameter.')
    th_range = np.linspace(0, 50, 100)

    num_iterations = len(th_range)
    logger.info('Running {} iterations'.format(num_iterations))

    tpr = []
    fpr = []

    best_th = -1
    max_score = 0
    for i, (th) in enumerate(th_range):
        logger.info('[{} of {}]\tvarThreshold={:.2f}'.format(i + 1, num_iterations, th))

        if '3.1' in cv.__version__:
            fgbg = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=50, decisionThreshold=th)
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorGMG(initializationFrames=50, decisionThreshold=th)
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        tpr_temp, fpr_temp, f1_score = evaluate_model(imageList, gtList, fgbg)

        # Store them in the array
        tpr.append(tpr_temp)
        fpr.append(fpr_temp)

        # Compare and select best parameters according to best score
        if f1_score > max_score:
            max_score = f1_score
            best_th = th

    area = plot_AUC_curve(tpr, fpr, cf.output_folder)

    logger.info('Finished search')
    logger.info('Best threshold: {:-3f}'.format(best_th))
    logger.info('Best F1-score: {:.3f}'.format(max_score))
    logger.info('AUC: {:.3f}'.format(area))

    if cf.save_results:
        if '3.1' in cv.__version__:
            fgbg = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=50, decisionThreshold=best_th)
        elif '2.4' in cv.__version__:
            fgbg = cv.BackgroundSubtractorGMG(initializationFrames=50, decisionThreshold=best_th)
        else:
            logger.error('OpenCV version not supported')
            sys.exit()

        for image in imageList:
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
            foreground = fgbg.apply(img)
            image_name = os.path.basename(image)
            image_name = os.path.splitext(image_name)[0]
            cv.imwrite(os.path.join(cf.results_path, 'GMG_' + image_name + '.' + cf.result_image_type), foreground)

def lbsp_background_estimator(imageList, gtList, cf):
    # Paper 'Background subtraction using local svd binary pattern' by L. Guo in 2016
    logger = logging.getLogger(__name__)
    logger.info('Running local svd binary pattern background estimation')

    lbsp = Subsense.LBSP()

    for image, gt in imageList:
        foreground = lbsp.apply(image)
        if cf.save_results:
            image_name = os.path.basename(image)
            image_name = os.path.splitext(image_name)[0]
            cv.imwrite(os.path.join(cf.results_path, 'LBSP_' + image_name + '.' + cf.result_image_type), foreground)

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
    elif cf.modelling_method == 'lbsp':
        lbsp_background_estimator(imageList, gtList, cf)
    else:
        logger.error('Modeling method not implemented')


# Entry point of the script
if __name__ == "__main__":
    main()