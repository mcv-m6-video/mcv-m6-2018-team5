from __future__ import division

import os
import time

import cv2 as cv
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as cl
import numpy as np
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import block_reduce
from sklearn.metrics import auc

from tools.optical_flow import read_kitti_flow

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8
CAR_COLOURS = [(0, 0, 255), (0, 106, 255), (0, 216, 255), (0, 255, 182), (0, 255, 76),
               (144, 255, 0), (255, 255, 0), (255, 148, 0), (255, 0, 178), (220, 0, 255)]


def aux_plot_auc_vs_pixels(auc_highway, pixels_range, output_folder=""):
    max_auc_highway = max(auc_highway)
    index_p = np.where(auc_highway == max_auc_highway)
    best_p_highway = pixels_range[index_p[0][0]]

    plt.title('Area Filtering - AUC vs P Pixels')
    plt.plot(pixels_range, auc_highway, label='AUC max =%.4f (P = %d)' % (max_auc_highway, best_p_highway))
    plt.ylabel('AUC')
    plt.xlabel('Number of Pixels')
    plt.ylim([0, 1])
    plt.xlim([pixels_range[0], pixels_range[-1]])
    leg = plt.legend(loc='lower right', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "week3_task_2_1_auc_pixels.png"))

    plt.show(block=False)
    plt.close()


def plot_auc_vs_pixels(auc_highway, auc_traffic, auc_fall, pixels_range, output_folder=""):
    max_auc_highway = max(auc_highway)
    index_p = np.where(auc_highway == max_auc_highway)
    best_p_highway = pixels_range[index_p[0][0]]

    max_auc_traffic = max(auc_traffic)
    index_p = np.where(auc_traffic == max_auc_traffic)
    best_p_traffic = pixels_range[index_p[0][0]]

    max_auc_fall = max(auc_fall)
    index_p = np.where(auc_fall == max_auc_fall)
    best_p_fall = pixels_range[index_p[0][0]]

    plt.title('Area Filtering - AUC vs P Pixels')
    plt.plot(pixels_range, auc_highway, label='AUC Highway max =%.4f (P = %d)' % (max_auc_highway, best_p_highway))
    plt.plot(pixels_range, auc_traffic, label='AUC Traffic max =%.4f (P = %d)' % (max_auc_traffic, best_p_traffic))
    plt.plot(pixels_range, auc_fall, label='AUC Fall max =%.4f (P = %d)' % (max_auc_fall, best_p_fall))
    plt.ylabel('AUC')
    plt.xlabel('Number of Pixels')
    plt.ylim([0, 1])
    plt.xlim([pixels_range[0], pixels_range[-1]])
    leg = plt.legend(loc='lower right', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "week3_task_2_1_auc_pixels.png"))

    plt.show(block=False)
    plt.close()


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


def plot_precision_recall_curve(precision, recall, output_folder="", color='blue'):
    plt.plot(recall, precision, color=color)
    try:
        auc_pr = auc(recall, precision, reorder=False)
    except ValueError:
        # Use reorder=True, even if it is not the correct way to compute the AUC for the PR curve
        auc_pr = auc(recall, precision, reorder=True)
    plt.fill_between(recall, 0, precision, color=color, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.title('Precision-Recall curve: AUC ={:.2f}'.format(auc_pr))
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "task_1_2_precision_recall.png"))
    plt.show(block=False)
    plt.close()

    return auc_pr


def plot_roc_curve(tpr, fpr, output_folder=""):
    area = auc(fpr, tpr, reorder=True)
    plt.step(fpr, tpr, color='b', alpha=0.2,
             where='post')

    plt.fill_between(fpr, 0, tpr, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([min(tpr), max(tpr)])
    plt.xlim([min(fpr), max(fpr)])
    plt.title('ROC curve: AUC={}'.format(area))
    if output_folder != "":
        plt.savefig(os.path.join(output_folder, "task_1_2_ROC.png"))
    plt.show(block=False)
    plt.close()
    return area


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


def plot_histogram_msen(msen, squared_errors, seq_name, output_path=None, nbins=50):
    plt.figure(figsize=(10, 5), dpi=200)
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

    if output_path is not None:
        plt.savefig(output_path)
    plt.close()


def plot_msen_image(image, squared_errors, pixel_errors, valid_pixels, seq_name, output_path=None):
    # Representation of errors as an image
    color_map = plt.cm.get_cmap('jet')
    im_data = cv.imread(image, cv.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 5), dpi=200)
    plt.imshow(im_data, cmap='gray')
    se_valid = np.zeros_like(squared_errors)
    se_valid[valid_pixels] = squared_errors[valid_pixels]
    se_valid *= pixel_errors
    plt.imshow(se_valid, cmap=color_map, alpha=0.5, label='Squared Errors')
    plt.title('Sequence {}'.format(seq_name))
    plt.colorbar(orientation="horizontal")
    plt.xlabel('Squared Error (Non-occluded areas)')
    plt.show(block=False)
    if output_path is not None:
        plt.savefig(output_path)
    plt.close()


def plot_optical_flow_hsv(img_path, vector_field_path, sequence_name, output_path, is_ndarray=False):
    if not is_ndarray:
        # Get the original image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        # Get the optical flow image
        optical_flow, _ = read_kitti_flow(cv.imread(vector_field_path, cv.IMREAD_UNCHANGED))
    else:
        img = img_path
        optical_flow = vector_field_path

    hsv = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 1] = 255
    magnitude, angle = cv.cartToPolar((optical_flow[:, :, 0]), (optical_flow[:, :, 1]))
    hsv[:, :, 0] = angle * 180 / np.pi
    hsv[:, :, 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    plt.figure(figsize=(10, 5), dpi=200)
    plt.imshow(img, cmap='gray')
    plt.imshow(bgr, alpha=0.5)
    plt.axis('off')
    plt.title(sequence_name)
    plt.show(block=False)
    plt.savefig(output_path)
    plt.close()


def plot_optical_flow(img_path, vector_field_path, downsample_factor, sequence_name, output_path, is_ndarray=False):
    if not is_ndarray:
        # Get the original image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        # Get the optical flow image
        optical_flow, _ = read_kitti_flow(cv.imread(vector_field_path, cv.IMREAD_UNCHANGED))
    else:
        img = img_path
        optical_flow = vector_field_path

    # Downsample optical flow image
    optical_flow_ds = block_reduce(optical_flow, block_size=(downsample_factor, downsample_factor, 1), func=np.mean)
    x_pos = np.arange(0, img.shape[1], downsample_factor)
    y_pos = np.arange(0, img.shape[0], downsample_factor)
    X = np.meshgrid(x_pos)
    Y = np.meshgrid(y_pos)

    plt.figure(figsize=(10, 5), dpi=200)
    plt.imshow(img, cmap='gray')
    plt.quiver(X, Y, optical_flow_ds[:, :, 0], optical_flow_ds[:, :, 1], color='yellow')
    plt.axis('off')
    plt.title(sequence_name)
    plt.savefig(output_path)
    plt.close()


def plot_adaptive_gaussian_grid_search(score_grid, alpha_range, rho_range, best_parameters, best_score, sequence_name,
                                       metric):
    x, y = np.meshgrid(rho_range, alpha_range)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(sequence_name + ' sequence')
    surf = ax.plot_surface(x, y, score_grid, cmap='viridis', vmin=0, vmax=1)
    # Best param
    ax.text(best_parameters['rho'], best_parameters['alpha'], best_score,
            'alpha={alpha:.2f},rho={rho:.2f}\n{metric}={f1:2f}'.format(metric=metric, f1=best_score, **best_parameters),
            None)
    ax.scatter(best_parameters['rho'], best_parameters['alpha'], best_score, marker='x', color='black')
    fig.colorbar(surf, ax=ax, fraction=0.1)
    ax.set_xlabel('rho')
    ax.set_ylabel('alpha')
    ax.set_zlabel('f1-score')
    fig.tight_layout()
    plt.show()
    plt.close()


def plot_back_evolution(backList, first_image, output_path):
    pixel_pos = [130, 130]
    mean1, var1 = plot_pixel_evolution(backList, first_image, pixel_pos, 'r', output_path)

    pixel_pos = [50, 50]
    mean2, var2 = plot_pixel_evolution(backList, first_image, pixel_pos, 'm', output_path)

    pixel_pos = [220, 300]
    mean3, var3 = plot_pixel_evolution(backList, first_image, pixel_pos, 'b', output_path)

    plt.figure()
    x = np.linspace(0, 255, 256)
    sigma1 = np.sqrt(var1)
    plt.plot(x, mlab.normpdf(x, mean1, sigma1), 'r-')
    sigma2 = np.sqrt(var2)
    plt.plot(x, mlab.normpdf(x, mean2, sigma2), 'm-')
    sigma3 = np.sqrt(var3)
    plt.plot(x, mlab.normpdf(x, mean3, sigma3), 'b-')
    plt.show()
    plt.savefig(os.path.join(output_path, 'gaussian.png'))


def plot_pixel_evolution(backList, first_image, pixel_pos, color, output_path):
    fig = plt.figure()
    ax_im = fig.add_subplot(211)

    back_img = cv.imread(backList[0], cv.IMREAD_GRAYSCALE)
    im = ax_im.imshow(back_img, cmap='gray')
    ax_im.set_axis_off()

    pixel = patches.Circle((pixel_pos[1], pixel_pos[0]), 5, linewidth=1, edgecolor=color, facecolor='none')
    ax_im.add_patch(pixel)

    ax = fig.add_subplot(212)
    ax.set_xlim(int(first_image), int(first_image) + len(backList))
    ax.set_ylim(0, 260)

    frames = []
    gray = []
    mean_val = 0
    mean = []
    var_val = 0
    std = []

    gray_line, = ax.plot(frames, gray, label='Gray level')
    mean_line, = ax.plot(frames, mean, label='Mean')
    std_line, = ax.plot(frames, std, label='Standard deviation')
    ax.legend()
    if color == 'r':
        ax.set_title('Red pixel')
    elif color == 'g':
        ax.set_title('Green pixel')
    elif color == 'b':
        ax.set_title('Blue pixel')
    ax.set_xlabel('Frame')

    plt.show(block=False)

    for count in range(0, len(backList)):
        fig.set_size_inches(8, 6, forward=True)

        back_img = cv.imread(backList[count], cv.IMREAD_GRAYSCALE)
        im.set_data(back_img)

        frames.append(int(first_image) + count)

        gray.append(back_img[pixel_pos[0], pixel_pos[1]])
        mean_val = (mean_val * (count + 1) + back_img[pixel_pos[0], pixel_pos[1]]) / (count + 2)
        mean.append(mean_val)
        var_val = (var_val * (count + 1) + np.square(back_img[pixel_pos[0], pixel_pos[1]] - mean_val)) / (count + 2)
        std.append(np.sqrt(var_val))
        gray_line.set_xdata(frames)
        gray_line.set_ydata(gray)
        mean_line.set_xdata(frames)
        mean_line.set_ydata(mean)
        std_line.set_xdata(frames)
        std_line.set_ydata(std)

        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.1)
        filename = 'pixel' + str(pixel_pos[0]) + '_' + str(pixel_pos[1]) + '_frame' + str(
            int(first_image) + count) + '.png'
        plt.savefig(os.path.join(output_path, filename))

    return mean_val, var_val


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def plot_optical_flow_histogram(optical_flow, area_search, output_path):
    # Compute 2D histogram of optical flow directions (x, y)
    xedges = np.arange(-area_search, area_search + 1)
    yedges = np.arange(-area_search, area_search + 1)
    flow_hist2d, _, _ = np.histogram2d(
        np.ravel(optical_flow[:, :, 0]), np.ravel(optical_flow[:, :, 1]), bins=(xedges, yedges)
    )
    flow_hist2d = flow_hist2d.T

    fig = plt.figure(figsize=(7, 7), dpi=200)
    ax = fig.add_subplot(111, title='2D histogram of optical flow directions',
                         aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    im.set_data(xcenters, ycenters, flow_hist2d)
    ax.images.append(im)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    plt.savefig(output_path)
    plt.close()


def draw_lines(img, lines, color=(255, 0, 0), thickness=3):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    plt.imshow(img)
    plt.show()


def show_detections(image_data, labeled_image, region_properties, save_path):
    N, M = image_data.shape
    fig, ax = plt.subplots(1)
    ax.imshow(image_data, cmap='gray')
    ax.imshow(labeled_image, alpha=0.3)
    for reg_prop in region_properties:
        min_row, min_col, max_row, max_col = reg_prop.bbox
        width, height = max_col - min_col, max_row - min_row
        rect = patches.Rectangle((min_col, min_row), width, height, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        centroid_row, centroid_col = reg_prop.centroid
        ax.scatter(centroid_col, centroid_row)
    plt.axis('off')
    plt.show(block=False)
    plt.savefig(save_path)
    plt.close()


def display_tracking_results(img, tracks, foreground, save_path):
    foreground = 255 * foreground.astype('uint8')
    foreground = cv.cvtColor(foreground, cv.COLOR_GRAY2BGR)
    cv.addWeighted(img, 0.7, foreground, 0.3, 0.0, img)
    if tracks != list():

        # Display the objects. If an object has not been detected
        # in this frame, display its predicted bounding box.
        if tracks != list():
            for track in tracks:
                car_colour = CAR_COLOURS[track.id % len(CAR_COLOURS)]
                for point in track.positions:
                    cv.circle(img, (int(point[0]), int(point[1])), 2, car_colour, 1)
                cv.polylines(img, [np.int32(track.positions)], False, car_colour, 1)

                for point in track.predictions:
                    cv.rectangle(img, (int(point[0]) - 2, int(point[1]) - 2), (int(point[0]) + 2, int(point[1]) + 2),
                                 car_colour, 1)
                cv.polylines(img, [np.int32(track.predictions)], False, car_colour, 1)

    cv.imwrite(save_path, img)


def display_current_speed_results(img, tracks, foreground, save_path):
    foreground = 255 * foreground.astype('uint8')
    foreground = cv.cvtColor(foreground, cv.COLOR_GRAY2BGR)
    cv.addWeighted(img, 0.7, foreground, 0.3, 0.0, img)
    if tracks != list():

        # Display the objects. If an object has not been detected
        # in this frame, display its predicted bounding box.
        if tracks != list():
            for track in tracks:
                car_colour = CAR_COLOURS[track.id % len(CAR_COLOURS)]
                cv.rectangle(img, (track.bbox[0], track.bbox[1]),
                             (track.bbox[0] + track.bbox[2], track.bbox[1] + track.bbox[3]), car_colour, 2)
                cv.rectangle(img, (track.bbox[0], track.bbox[1]), (track.bbox[0] + 20, track.bbox[1] + 10), car_colour,
                             -1)
                # cv.putText(img, str(track.id), (track.bbox[0], track.bbox[1]+10), cv.FONT_HERSHEY_PLAIN, 1,
                # (0, 0, 0), 1)
                cv.putText(img, str(np.round(track.current_speed)), (track.bbox[0], track.bbox[1] + 10),
                           cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),
                           1)

    cv.imwrite(save_path, img)


def display_speed_results(img, tracks, max_speed, lanes, save_path, roi, margin):
    # Display the objects. If an object has not been detected
    # in this frame, display its predicted bounding box.
    if tracks != list():
        for track in tracks:
            if track.current_speed > 0:
                if track.current_speed > 1.1*max_speed:
                    color = (0, 0, 255)
                elif track.current_speed < max_speed * 0.8:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                cv.rectangle(img, (track.bbox[0], track.bbox[1]),
                             (track.bbox[0] + track.bbox[2], track.bbox[1] + track.bbox[3]), color, 2)
                cv.rectangle(img, (track.bbox[0], track.bbox[1]), (track.bbox[0] + 20, track.bbox[1] + 10), color, -1)
                cv.putText(img, str(np.round(track.current_speed)), (track.bbox[0], track.bbox[1] + 10),
                           cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),
                           1)
    cv.line(img, (roi[0][0], roi[0][1]), (roi[1][0], roi[1][1]), (255, 255, 0), 2)
    cv.line(img, (roi[2][0], roi[2][1]), (roi[3][0], roi[3][1]), (255, 255, 0), 2)
    copy = cv.copyMakeBorder(img, 0, 75, margin, margin, cv.BORDER_CONSTANT)

    for n, lane in enumerate(lanes):

        if lane.sum_vehicles > 0:
            lane_current_velocity = lane.sum_velocities / lane.sum_vehicles
        else:
            lane_current_velocity = 0

        cv.putText(copy, 'Lane {}:'.format(n + 1), (150 * n + 2, img.shape[0] + 15),
                   cv.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
        cv.putText(copy, '  {:.0f} vehicles'.format(lane.total_vehicles), (150 * n + 2, img.shape[0] + 35),
                   cv.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
        cv.putText(copy, '  {} density'.format(lane.density), (150 * n + 2, img.shape[0] + 50),
                   cv.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
        cv.putText(copy, '  {:.0f} km/h'.format(lane_current_velocity), (150 * n + 2, img.shape[0] + 65),
                   cv.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

    cv.imwrite(save_path, copy)


def visualize_lanes(img, lanes, save_path):
    plt.imshow(img)
    colors = cl.BASE_COLORS.values()
    for idx in range(len(lanes)):
        pol = patches.Polygon(lanes[idx], alpha=0.4, facecolor=colors[idx])
        plt.gca().add_artist(pol)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


def visualize_roi(img, roi, save_path):
    plt.imshow(img)
    colors = cl.BASE_COLORS.values()
    pol = patches.Polygon(roi, alpha=0.4, facecolor=colors[6])
    plt.gca().add_artist(pol)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
