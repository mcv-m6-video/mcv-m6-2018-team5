from __future__ import division

import logging

import numpy as np
import cv2 as cv
from skimage import morphology
from metrics import segmentation_metrics
from tools import background_modeling
from sklearn import metrics

EPSILON = 1e-8


def hole_filling(image, four_connectivity=True):
    if four_connectivity:
        output_image = morphology.remove_small_holes(image, connectivity=1)
    else:
        output_image = morphology.remove_small_holes(image, connectivity=2)

    return output_image


# Remove small regios with less than n pixels
def remove_small_regions(image, nr_pixels, conn_pixels=True):
    # The connectivity defining the neighborhood of a pixel
    if conn_pixels:
        conn = 1
    else:
        conn = 2
    output_image = morphology.remove_small_objects(image, min_size=nr_pixels, connectivity=conn)

    return output_image


def area_filtering_auc_vs_pixels(cf, background_img_list, foreground_img_list, foreground_gt_list):
    logger = logging.getLogger(__name__)

    mean, variance = background_modeling.multivariative_gaussian_modelling(background_img_list,
                                                                           cf.color_space)

    alpha_range = np.linspace(cf.evaluate_alpha_range[0], cf.evaluate_alpha_range[1],
                              num=cf.evaluate_alpha_values)

    auc = []  # Store AUC values to plot
    pixels_range = np.linspace(cf.P_pixels_range[0], cf.P_pixels_range[1], num=cf.P_pixels_values)
    best_alphas = []
    for pixels in pixels_range:
        logger.info("Pixels P: " + str(pixels))

        precision = []
        recall = []
        f1_score = []
        logger.info('Iterating alpha from {} to {}, {} steps'.format(alpha_range[0], alpha_range[-1], len(alpha_range)))
        for alpha in alpha_range:
            tp = 0
            fp = 0
            tn = 0
            fn = 0

            for (image, gt) in zip(foreground_img_list, foreground_gt_list):
                gt_img = cv.imread(gt, cv.IMREAD_GRAYSCALE)
                foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
                    image, mean, variance, alpha, cf.rho, cf.color_space
                )
                foreground = hole_filling(foreground, cf.four_connectivity)

                # Area Filtering
                foreground = remove_small_regions(foreground, pixels, cf.four_connectivity)
                foreground = np.array(foreground, dtype='uint8')
                tp_temp, fp_temp, tn_temp, fn_temp = segmentation_metrics.evaluate_single_image(foreground,
                                                                                                gt_img)

                tp += tp_temp
                fp += fp_temp
                tn += tn_temp
                fn += fn_temp

            pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * pre * rec / (pre + rec + EPSILON)

            precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            f1_score.append(f1)

        best_f1_score = max(f1_score)
        index_alpha = f1_score.index(best_f1_score)
        best_alpha = alpha_range[index_alpha]
        best_alphas.append(best_alpha)

        try:
            auc_pr = metrics.auc(recall, precision, reorder=False)
        except ValueError:
            # Use reorder=True, even if it is not the correct way to compute the AUC for the PR curve
            auc_pr = metrics.auc(recall, precision, reorder=True)

        logger.info('Best alpha: {:.4f}'.format(best_alpha))
        logger.info('Best F1-score: {:.4f}'.format(best_f1_score))
        logger.info('AUC: {:.4f}'.format(auc_pr))
        auc.append(auc_pr)

    max_auc = max(auc)
    index_pixels = auc.index(max_auc)
    best_pixels = pixels_range[index_pixels]
    best_alpha = best_alphas[index_pixels]
    logger.info('Best AUC: {:.4f}'.format(max_auc))

    return auc, pixels_range, best_pixels, best_alpha
