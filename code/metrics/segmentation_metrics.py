import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

def evaluate(testList, gtList):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    num_images = len(testList)
    for test_image,gt_image in zip(testList,gtList):
        img = cv.imread(test_image, cv.IMREAD_GRAYSCALE)
        gt_img = cv.imread(gt_image, cv.IMREAD_GRAYSCALE)
        ret, gt_img = cv.threshold(gt_img, 150, 1, cv.THRESH_BINARY)
        h, w = np.shape(img)
        for i in range(0, h):
            for j in range(0, w):
                if img[i,j] == 1:
                    if img[i,j] == gt_img[i,j]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if img[i,j] == gt_img[i,j]:
                        TN += 1
                    else:
                        FN += 1

    print("TP: "+str(TP))
    print("FP" + str(FP))
    print("TN: "+str(TN))
    print("FN" + str(FN))

    precision = (TP / float(TP + FP))
    recall = TP / float(TP + FN)
    F1_score = 2*precision*recall / (precision + recall)
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

        precision = (TP_temp / float(TP_temp + FP))
        recall = TP_temp / float(TP_temp + FN)

        TP.append(TP_temp)
        T.append(TP_temp + FN)
        F1_score.append(2 * precision * recall / (precision + recall))

    return TP,T,F1_score

def desynchronization(testList, gtList, frames):

    for frame in frames:
        F1_score = []
        gtList_des = list(gtList)

        if frame != 0:
            del gtList_des[0:frame-1]

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

            precision = (TP / float(TP + FP))
            recall = TP / float(TP + FN)

            F1_score.append(2 * precision * recall / (precision + recall))

        plt.plot(F1_score, label=str(frame) + ' de-synchronization frames')

    plt.xlabel('time')
    plt.legend(loc='upper right', fontsize='medium')
    plt.show()
