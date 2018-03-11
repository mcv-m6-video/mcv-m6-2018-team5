import numpy as np
import cv2 as cv
from skimage import morphology
from metrics import segmentation_metrics
from tools import background_modeling
from sklearn.metrics import auc

EPSILON = 1e-8

def hole_filling(image, fourConnectivity = True):
    if fourConnectivity:
        outputImage = morphology.remove_small_holes(image, connectivity=1)
    else:
        outputImage = morphology.remove_small_holes(image, connectivity=2)

    return outputImage
# Remove small regios with less than n pixels
def remove_small_regions(image, nr_pixels, conn_pixels = True):

    # The connectivity defining the neighborhood of a pixel
    if conn_pixels:
        conn = 1
    else:
        conn = 2
    outputImage = morphology.remove_small_objects(image, min_size=nr_pixels, connectivity=conn)

    return outputImage


def area_filtering_AUC_vs_pixels(cf, logger, background_img_list, foreground_img_list,
                                 foreground_gt_list):

    mean, variance = background_modeling.multivariative_gaussian_modelling(background_img_list,
                                                                           cf.color_space)

    alpha_range = np.linspace(cf.evaluate_alpha_range[0], cf.evaluate_alpha_range[1],
                              num=cf.evaluate_alpha_values)

    AUC = [] # Store AUC values to plot
    pixels_range = np.linspace(cf.P_pixels_range[0], cf.P_pixels_range[1], num=30)
    best_alphas = []
    for pixels in pixels_range:
        logger.info("Pixels P: " + str(pixels))

        precision = []
        recall = []
        F1_score = []
        for alpha in alpha_range:
            logger.info("Alpha : " + str(alpha))
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
            F1_score.append(f1)


        best_f1_score = max(F1_score)
        index_alpha = F1_score.index(best_f1_score)
        best_alpha = alpha_range[index_alpha]
        best_alphas.append(best_alpha)

        try:
            auc_pr = auc(recall, precision, reorder=False)
        except ValueError:
            # Use reorder=True, even if it is not the correct way to compute the AUC for the PR curve
            auc_pr = auc(recall, precision, reorder=True)

        logger.info('Best alpha: {:.4f}'.format(best_alpha))
        logger.info('Best F1-score: {:.4f}'.format(best_f1_score))
        logger.info('AUC: {:.4f}'.format(auc_pr))
        AUC.append(auc_pr)

    max_AUC = max(AUC)
    index_pixels = AUC.index(max_AUC)
    best_pixels = pixels_range[index_pixels]
    best_alpha = best_alphas[index_pixels]
    logger.info('Best AUC: {:.4f}'.format(max_AUC))

    return AUC, pixels_range, best_pixels, best_alpha
