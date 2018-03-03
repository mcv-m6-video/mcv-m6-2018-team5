from __future__ import division

import os
import sys
import logging
import time

import cv2 as cv
import numpy as np

from tools.background_modeling import *

EPSILON = 1e-8


def evaluate_single_image(test_img, gt_img):
    TP = np.count_nonzero((test_img == 1) & (gt_img == 255))
    FP = np.count_nonzero((test_img == 1) & ((gt_img == 0) | (gt_img == 50)))
    TN = np.count_nonzero((test_img == 0) & ((gt_img == 0) | (gt_img == 50)))
    FN = np.count_nonzero((test_img == 0) & (gt_img == 255))
    return TP, FP, TN, FN


def evaluate_foreground_estimation(modelling_method, imageList, gtList, mean, variance, alpha=(1,),
                                   rho=0.5):

    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    precision_list = []
    recall_list = []
    F1_score_list = []

    if modelling_method == 'mog':
        fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    elif modelling_method == 'mog2':
        fgbg = cv.createBackgroundSubtractorMOG2()
    elif modelling_method == 'gmg':
        fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
    elif modelling_method == 'lsbp':
        fgbg = cv.bgsegm.createBackgroundSubtractorLSBP()

    for al in alpha:
        metrics = np.zeros(4)
        for test_image, gt_image in zip(imageList, gtList):
            if modelling_method == 'gaussian':
                foreground = foreground_estimation(test_image, mean, variance, al)
            elif modelling_method == 'adaptive':
                foreground = adaptive_foreground_estimation(test_image, mean, variance, alpha, rho)
            elif modelling_method == 'mog':
                foreground, fgbg = mog_foreground_estimation(test_image, fgbg)
            elif modelling_method == 'mog2':
                foreground, fgbg = mog_foreground_estimation(test_image, fgbg)
            elif modelling_method == 'gmg':
                foreground, fgbg = mog_foreground_estimation(test_image, fgbg)
            elif modelling_method == 'lsbp':
                foreground, fgbg = mog_foreground_estimation(test_image, fgbg)
            foreground = np.array(foreground, dtype='uint8')
            gt_img = cv.imread(gt_image, cv.IMREAD_GRAYSCALE)
            metrics += evaluate_single_image(foreground, gt_img)

        TP, FP, TN, FN = metrics

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1_score = 2 * precision * recall / (precision + recall + EPSILON)

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

        precision_list.append(precision)
        recall_list.append(recall)
        F1_score_list.append(F1_score)

    return TP_list, TN_list, FP_list, FN_list, precision_list, recall_list, F1_score_list

def evaluate(testList, gtList):
    logger = logging.getLogger(__name__)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    start = time.time()
    for test_image, gt_image in zip(testList, gtList):
        img = cv.imread(test_image, cv.IMREAD_GRAYSCALE)
        gt_img = cv.imread(gt_image, cv.IMREAD_GRAYSCALE)

        TP += np.count_nonzero((img == 1) & (gt_img == 255))
        FP += np.count_nonzero((img == 1) & ((gt_img == 0) | (gt_img == 50)))
        TN += np.count_nonzero((img == 0) & ((gt_img == 0) | (gt_img == 50)))
        FN += np.count_nonzero((img == 0) & (gt_img == 255))

    end = time.time()

    logger.info("Time to process {} images: {:.2f} s".format(len(testList), end - start))
    logger.info("TP: " + str(TP))
    logger.info("FP: " + str(FP))
    logger.info("TN: " + str(TN))
    logger.info("FN: " + str(FN))

    total_predictions = TP + FP
    total_true = TP + FN

    precision = (TP / total_predictions) if total_predictions > 0 else 0.0
    recall = TP / total_true if total_true > 0 else 0.0
    F1_score = 2 * precision * recall / (precision + recall + EPSILON)
    return precision, recall, F1_score


def temporal_evaluation(testList, gtList):
    TP = []
    T = []
    F1_score = []
    for test_image, gt_image in zip(testList, gtList):
        img = cv.imread(test_image, cv.IMREAD_GRAYSCALE)
        gt_img = cv.imread(gt_image, cv.IMREAD_GRAYSCALE)

        true_pos = np.count_nonzero((img == 1) & (gt_img == 255))
        false_pos = np.count_nonzero((img == 1) & ((gt_img == 0) | (gt_img == 50)))
        false_neg = np.count_nonzero((img == 0) & (gt_img == 255))

        total_predictions = true_pos + false_pos
        total_true = true_pos + false_neg

        precision = (true_pos / total_predictions) if total_predictions > 0 else 0.0
        recall = (true_pos / total_true) if total_true > 0 else 0.0

        TP.append(true_pos)
        T.append(total_true)
        F1_score.append(2 * precision * recall / (precision + recall + EPSILON))

    return TP, T, F1_score


def desynchronization(testList, gtList, frames):
    num_desynch = 0
    num_image = 0
    F1_score = np.zeros((len(frames), len(testList)))
    for frame in frames:
        # F1_score = []
        gtList_des = list(gtList)
        if frame != 0:
            del gtList_des[0:frame - 1]

        for test_image, gt_image in zip(testList, gtList_des):
            img = cv.imread(test_image, cv.IMREAD_GRAYSCALE)
            gt_img = cv.imread(gt_image, cv.IMREAD_GRAYSCALE)

            true_pos = np.count_nonzero((img == 1) & (gt_img == 255))
            false_pos = np.count_nonzero((img == 1) & ((gt_img == 0) | (gt_img == 50)))
            false_neg = np.count_nonzero((img == 0) & (gt_img == 255))

            total_predictions = true_pos + false_pos
            total_true = true_pos + false_neg

            precision = (true_pos / total_predictions) if total_predictions > 0 else 0.0
            recall = (true_pos / total_true) if total_true > 0 else 0.0

            F1_score[num_desynch][num_image] = 2 * precision * recall / (precision + recall + EPSILON)
            num_image += 1

        num_desynch += 1
        num_image = 0

    return F1_score
