import argparse
import logging
import logging.handlers
import sys

import cv2
import numpy as np

from tools import background_modeling, foreground_improving
from tools.image_parser import get_image_list_changedetection_dataset
from tools.tracking_sota import VehicleCounter
from utils.load_configutation import Configuration

# ============================================================================

IMAGE_DIR = "images"
IMAGE_FILENAME_FORMAT = IMAGE_DIR + "/frame_%04d.png"

# Support either video file or individual frames
CAPTURE_FROM_VIDEO = False
if CAPTURE_FROM_VIDEO:
    IMAGE_SOURCE = "traffic.avi"  # Video file
else:
    IMAGE_SOURCE = IMAGE_FILENAME_FORMAT  # Image sequence

# Time to wait between frames, 0=forever
WAIT_TIME = 1  # 250 # ms

LOG_TO_FILE = True

# Colours for drawing on processed frames
DIVIDER_COLOUR = (255, 255, 0)
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)


# ============================================================================

def init_logging():
    main_logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S')

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if LOG_TO_FILE:
        handler_file = logging.handlers.RotatingFileHandler("debug.log"
                                                            , maxBytes=2 ** 24
                                                            , backupCount=10)
        handler_file.setFormatter(formatter)
        main_logger.addHandler(handler_file)

    main_logger.setLevel(logging.DEBUG)

    return main_logger


# ============================================================================

def save_frame(file_name_format, frame_number, frame, label_format):
    file_name = file_name_format % frame_number
    label = label_format % frame_number

    log.debug("Saving %s as '%s'", label, file_name)
    cv2.imwrite(file_name, frame)


# ============================================================================

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)


# ============================================================================

def detect_vehicles(fg_mask, cf):
    log = logging.getLogger("detect_vehicles")

    MIN_CONTOUR_WIDTH = cf.min_width
    MIN_CONTOUR_HEIGHT = cf.min_height

    # Find the contours of any vehicles in the image
    contours, hierarchy = cv2.findContours(fg_mask
                                           , cv2.RETR_EXTERNAL
                                           , cv2.CHAIN_APPROX_SIMPLE)

    log.debug("Found %d vehicle contours.", len(contours))

    matches = []
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)

        log.debug("Contour #%d: pos=(x=%d, y=%d) size=(w=%d, h=%d) valid=%s"
                  , i, x, y, w, h, contour_valid)

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid))

    return matches


# ============================================================================

def filter_mask(fg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Fill any small holes
    closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)

    return dilation


# ============================================================================

def process_frame(frame_number, frame, fg_mask, car_counter, cf):
    log = logging.getLogger("process_frame")

    # Create a copy of source frame to draw into
    processed = frame.copy()

    # Draw dividing line -- we count cars as they cross this line.
    cv2.line(processed, (0, car_counter.divider), (frame.shape[1], car_counter.divider), DIVIDER_COLOUR, 1)

    # Remove the background
    fg_mask = filter_mask(fg_mask)

    save_frame(IMAGE_DIR + "/mask_%04d.png"
               , frame_number, fg_mask, "foreground mask for frame #%d")

    matches = detect_vehicles(fg_mask, cf)

    log.debug("Found %d valid vehicle contours.", len(matches))
    for (i, match) in enumerate(matches):
        contour, centroid = match

        log.debug("Valid vehicle contour #%d: centroid=%s, bounding_box=%s", i, centroid, contour)

        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)

    log.debug("Updating vehicle count...")
    car_counter.update_count(matches, processed)

    return processed


# ============================================================================

def main():
    log = logging.getLogger("main")
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='W5 - Vehicle tracker and speed estimator [Team 5]')
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

    log.debug("Creating background subtractor...")

    image_list = get_image_list_changedetection_dataset(
        cf.dataset_path, 'in', cf.first_image, cf.image_type, cf.nr_images
    )
    background_img_list = image_list[:len(image_list) // 2]
    foreground_img_list = image_list[(len(image_list) // 2):]
    log.debug("Pre-training the background subtractor...")
    mean, variance = background_modeling.multivariative_gaussian_modelling(background_img_list, cf.color_space)

    car_counter = None  # Will be created after first frame is captured

    frame_width = cv2.imread(background_img_list[0]).shape[1]
    frame_height = cv2.imread(background_img_list[0]).shape[0]
    log.debug("Video capture frame size=(w=%d, h=%d)", frame_width, frame_height)

    log.debug("Starting capture loop...")
    frame_number = -1
    for image_path in foreground_img_list:

        frame_number += 1
        log.debug("Capturing frame #%d...", frame_number)
        frame = cv2.imread(image_path)
        log.debug("Got frame #%d: shape=%s", frame_number, frame.shape)
        foreground, mean, variance = background_modeling.adaptive_foreground_estimation_color(
            image_path, mean, variance, cf.alpha, cf.rho, cf.color_space)

        foreground = foreground_improving.hole_filling(foreground, cf.four_connectivity)
        foreground = foreground_improving.remove_small_regions(foreground, cf.area_filtering_P)
        foreground = foreground_improving.image_opening(foreground, cf.opening_strel, cf.opening_strel_size)
        foreground = foreground_improving.image_closing(foreground, cf.closing_strel, cf.closing_strel_size)
        foreground = foreground_improving.remove_small_regions(foreground, cf.area_filtering_P_post)

        if car_counter is None:
            # We do this here, so that we can initialize with actual frame size
            log.debug("Creating vehicle counter...")
            car_counter = VehicleCounter(frame.shape[:2], frame.shape[0] / 2)

        # Archive raw frames from video to disk for later inspection/testing
        if CAPTURE_FROM_VIDEO:
            save_frame(IMAGE_FILENAME_FORMAT
                       , frame_number, frame, "source frame #%d")

        log.debug("Processing frame #%d...", frame_number)
        foreground_ = np.array(foreground, dtype=np.uint8)
        processed = process_frame(frame_number, frame, foreground_, car_counter, cf)

        save_frame(cf.output_folder + "/processed_%04d.png"
                   , frame_number, processed, "processed frame #%d")

        cv2.imshow('Source Image', foreground.astype('uint8') * 255)
        cv2.imshow('Processed Image', processed)

        log.debug("Frame #%d processed.", frame_number)

        c = cv2.waitKey(WAIT_TIME)
        if c == 27:
            log.debug("ESC detected, stopping...")
            break

    log.debug("Closing video capture device...")
    cv2.destroyAllWindows()
    log.debug("Done.")


# ============================================================================

if __name__ == "__main__":
    log = init_logging()

    main()
