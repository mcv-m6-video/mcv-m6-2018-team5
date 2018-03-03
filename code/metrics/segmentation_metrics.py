from __future__ import division

import os
import sys
import logging
import cv2 as cv
import numpy as np
import time

from tools.background_modeling import foreground_estimation

EPSILON = 1e-8

def evaluate_single_image(test_img, gt_img):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    TP += np.count_nonzero((test_img == 1) & (gt_img == 255))
    FP += np.count_nonzero((test_img == 1) & ((gt_img == 0) | (gt_img == 50)))
    TN += np.count_nonzero((test_img == 0) & ((gt_img == 0) | (gt_img == 50)))
    FN += np.count_nonzero((test_img == 0) & (gt_img == 255))

    precision = (TP / TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1_score = 2 * precision * recall / (precision + recall + EPSILON)
    return TP, FP, TN, FN, F1_score

def evaluate_foreground_estimation(background, gt):
    TP, FP, TN, FN, F1_score = evaluate_single_image(background, gt)

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
