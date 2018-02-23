import cv2 as cv
import numpy as np
import sys

def evaluate(testList, gtList):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    count = 0

    for name in gtList:
        img = cv.imread(name, cv.IMREAD_GRAYSCALE)
        img = cv.normalize(img, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        gt_img = cv.imread(gtList[count], cv.IMREAD_GRAYSCALE)
        gt_img = cv.normalize(gt_img, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
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
        count += 1
    print("TP: "+str(TP))
    print("FP" + str(FP))
    print("TN: "+str(TN))
    print("FN" + str(FN))

    precision = (TP / float(TP + FP))
    recall = TP / float(TP + FN)
    F1_score = 2*precision*recall / (precision + recall)
    return precision, recall, F1_score