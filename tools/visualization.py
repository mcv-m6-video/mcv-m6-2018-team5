import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
import os
from skimage.measure import block_reduce
from metrics.optical_flow import read_flow_field
from sklearn.metrics import auc

def plot_metrics_vs_threshold(precision, recall, F1_score, threshold,
                              output_folder=""):
    plt.title('Precision, Recall and F1-score vs Threshold')
    plt.plot(threshold, precision, label="Precision")
    plt.plot(threshold, recall, label="Recall")
    plt.plot(threshold, F1_score, label="F1-score")
    plt.xlabel('Threshold')
    plt.ylabel('Metric')
    leg = plt.legend(loc='upper right', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "task_1_2_threshold.png"))
    plt.show(block=False)
    plt.close()


def plot_precision_recall_curve(precision, recall, output_folder=""):
    # precision.insert(0, precision[0])
    # precision.insert(len(precision), precision[-1])
    # recall.insert(0, 1)
    # recall.insert(len(recall), 0)
    area = auc(recall, precision)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    average_precision = reduce(lambda x, y: x + y, precision) / len(precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.ylim([0, 1])
    # plt.xlim([0, 1])
    plt.ylim([min(precision), max(precision)])
    plt.xlim([min(recall), max(recall)])
    plt.title('Precision-Recall curve: Avg Prec ={0:0.2f} AUC={1:0.3f}'.format(average_precision, area))
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "task_1_2_precision_recall.png"))
    plt.show(block=False)
    plt.close()

def plot_true_positives(TP, T, output_folder=""):
    plt.plot(TP, label='True Positives')
    plt.plot(T, label='Foreground pixels')
    plt.xlabel('time')
    plt.legend(loc='upper right', fontsize='medium')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show(block=False)
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "task_2_1.png"))
    plt.close()


def plot_F1_score(F1_score, output_folder=""):
    plt.plot(F1_score, label='F1 Score')
    plt.xlabel('time')
    plt.legend(loc='upper right', fontsize='medium')
    plt.show(block=False)
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "task_2_2.png"))
    plt.close()


def plot_desynch_vs_time(F1_score, desynchronization_frames, output_folder=""):
    for i in range(0, len(desynchronization_frames)):
        plt.plot(F1_score[i], label=str(desynchronization_frames[i]) + ' de-synchronization frames')

    plt.xlabel('time')
    plt.legend(loc='upper right', fontsize='medium')
    plt.show(block=False)

    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "task_4.png"))

    plt.close()


def plot_histogram_msen(msen, squared_errors, seq_name, output_folder="", nbins=50):
    fig = plt.figure()
    color_map = plt.cm.get_cmap('jet')
    # Histogram
    n, bins, patches = plt.hist(squared_errors, bins=nbins, normed=True)
    formatter = mticker.FuncFormatter(lambda v, pos: str(v * 100))
    plt.gca().yaxis.set_major_formatter(formatter)
    # scale values to interval [0,1]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', color_map(c))
    # Add colorbar
    norm = mpl.colors.Normalize(vmin=np.min(squared_errors), vmax=np.max(squared_errors))
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm._A = []
    plt.colorbar(sm)
    plt.axvline(msen, c='darkred', linestyle=':', label='Mean Squared Error')
    plt.xlabel('Squared Error (Non-occluded areas)')
    plt.ylabel('% of Pixels')
    plt.title('Sequence {}'.format(seq_name))
    plt.legend()
    plt.show(block=False)

    if output_folder != "":
        save_path = os.path.join(output_folder, "task_3_histogram_{}.png".format(seq_name))
        plt.savefig(save_path)
    plt.close()


def plot_msen_image(image, squared_errors, pixel_errors, valid_pixels, seq_name, output_folder=""):
    # Representation of errors as an image
    color_map = plt.cm.get_cmap('jet')
    im_data = cv.imread(image, cv.IMREAD_GRAYSCALE)
    plt.imshow(im_data, cmap='gray')
    se_valid = np.zeros_like(squared_errors)
    se_valid[valid_pixels] = squared_errors[valid_pixels]
    se_valid *= pixel_errors
    plt.imshow(se_valid, cmap=color_map, alpha=0.5, label='Squared Errors')
    plt.title('Sequence {}'.format(seq_name))
    plt.colorbar(orientation="horizontal")
    plt.xlabel('Squared Error (Non-occluded areas)')
    plt.show(block=False)
    save_path = os.path.join(output_folder, "task_3_error_image_{}.png".format(seq_name))
    plt.savefig(save_path)
    plt.close()


def plot_optical_flow_hsv(img_path, vector_field_path, sequence_name, output_path):
    # Get the original image
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # Get the optical flow image
    optical_flow, _ = read_flow_field(cv.imread(vector_field_path, cv.IMREAD_UNCHANGED))

    magnitude, angle = cv.cartToPolar(np.square(optical_flow[:, :, 0]), np.square(optical_flow[:, :, 1]),
                                      None, None, True)
    magnitude = cv.normalize(magnitude, 0, 255, norm_type=cv.NORM_MINMAX)

    optical_flow_hsv = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    optical_flow_hsv[..., 0] = angle
    optical_flow_hsv[..., 1] = 255
    optical_flow_hsv[:, :, 2] = magnitude
    # optical_flow_hsv[:, :, 1] = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    optical_flow_rgb = cv.cvtColor(optical_flow_hsv, cv.COLOR_HSV2BGR)

    plt.imshow(img, cmap='gray')
    plt.imshow(optical_flow_rgb, alpha=0.5)
    plt.axis('off')
    plt.title(sequence_name)
    plt.show(block=False)
    plt.savefig(output_path)
    plt.close()


def plot_optical_flow(img_path, vector_field_path, downsample_factor, sequence_name, output_path):
    # Get the original image
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # Get the optical flow image
    optical_flow, _ = read_flow_field(cv.imread(vector_field_path, cv.IMREAD_UNCHANGED))

    # Downsample optical flow image
    optical_flow_ds = block_reduce(optical_flow, block_size=(downsample_factor, downsample_factor, 1), func=np.mean)
    x_pos = np.arange(0, img.shape[1], downsample_factor)
    y_pos = np.arange(0, img.shape[0], downsample_factor)
    X = np.meshgrid(x_pos)
    Y = np.meshgrid(y_pos)

    plt.imshow(img, cmap='gray')
    plt.quiver(X, Y, optical_flow_ds[:, :, 0], optical_flow_ds[:, :, 1], color='yellow')
    plt.axis('off')
    plt.title(sequence_name)
    plt.show(block=False)
    plt.savefig(output_path)
    plt.close()
