import logging

import cv2 as cv
import numpy as np


def evaluate(testList, gtList):
    logger = logging.getLogger(__name__)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for test_image, gt_image in zip(testList, gtList):
        img = cv.imread(test_image, cv.IMREAD_GRAYSCALE)
        gt_img = cv.imread(gt_image, cv.IMREAD_GRAYSCALE)
        ret, gt_img = cv.threshold(gt_img, 150, 1, cv.THRESH_BINARY)
        h, w = np.shape(img)
        for i in range(0, h):
            for j in range(0, w):
                if img[i, j] == 1:
                    if img[i, j] == gt_img[i, j]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if img[i, j] == gt_img[i, j]:
                        TN += 1
                    else:
                        FN += 1

    logger.info("TP: " + str(TP))
    logger.info("FP: " + str(FP))
    logger.info("TN: " + str(TN))
    logger.info("FN: " + str(FN))

    precision = (TP / float(TP + FP))
    recall = TP / float(TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, F1_score


def temporal_evaluation(testList, gtList):
    TP = []
    T = []
    F1_score = []
    for test_image, gt_image in zip(testList, gtList):
        TP_temp = 0
        FP = 0
        TN = 0
        FN = 0
        img = cv.imread(test_image, cv.IMREAD_GRAYSCALE)
        gt_img = cv.imread(gt_image, cv.IMREAD_GRAYSCALE)
        ret, gt_img = cv.threshold(gt_img, 150, 1, cv.THRESH_BINARY)
        h, w = np.shape(img)
        for i in range(0, h):
            for j in range(0, w):
                if img[i, j] == 1:
                    if img[i, j] == gt_img[i, j]:
                        TP_temp += 1
                    else:
                        FP += 1
                else:
                    if img[i, j] == gt_img[i, j]:
                        TN += 1
                    else:
                        FN += 1

        if TP_temp != 0 or FP != 0:
            precision = (TP_temp / float(TP_temp + FP))
        else:
            precision = 0
        if TP != 0 or FN != 0:
            recall = TP_temp / float(TP_temp + FN)
        else:
            recall = 0

        TP.append(TP_temp)
        T.append(TP_temp + FN)

        if precision != 0 or recall != 0:
            F1_score.append(2 * precision * recall / (precision + recall))
        else:
            F1_score.append(0)

    return TP, T, F1_score


def desynchronization(testList, gtList, frames):
    num_desynch = 0
    num_image = 0
    F1_score = np.empty((len(frames), len(testList)))
    for frame in frames:
        # F1_score = []
        gtList_des = list(gtList)
        if frame != 0:
            del gtList_des[0:frame - 1]

        for test_image, gt_image in zip(testList, gtList_des):
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            img = cv.imread(test_image, cv.IMREAD_GRAYSCALE)
            gt_img = cv.imread(gt_image, cv.IMREAD_GRAYSCALE)
            ret, gt_img = cv.threshold(gt_img, 150, 1, cv.THRESH_BINARY)
            h, w = np.shape(img)
            for i in range(0, h):
                for j in range(0, w):
                    if img[i, j] == 1:
                        if img[i, j] == gt_img[i, j]:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if img[i, j] == gt_img[i, j]:
                            TN += 1
                        else:
                            FN += 1

            if TP != 0 or FP != 0:
                precision = (TP / float(TP + FP))
            else:
                precision = 0

            if TP != 0 or FN != 0:
                recall = TP / float(TP + FN)
            else:
                recall = 0

            if precision != 0 or recall != 0:
                F1_score[num_desynch][num_image] = 2 * precision * recall / (precision + recall)
            else:
                F1_score[num_desynch][num_image] = 0
            num_image += 1

        num_desynch += 1
        num_image = 0

    return F1_score
