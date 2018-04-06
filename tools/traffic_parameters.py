import numpy as np
import cv2 as cv
from visualization import draw_lines
from matplotlib import path


def speed_estimation(vehicle, pix_met, frame_sec, dt=1):

    current = len(vehicle.positions) - 1
    if current >= dt:
        dist = (np.sqrt(np.sum(
            np.square(np.asarray(vehicle.positions[current]) - np.asarray(vehicle.positions[0]))))) * (
               1 / pix_met)
        time = len(vehicle.positions) * (1 / frame_sec)
        speed = (dist / time * 3600) / 1000
        vehicle.speed = speed


def is_inside_speed_roi(position, speed_roi):

    polygon = path.Path(np.asarray(speed_roi))

    return polygon.contains_point(position)


def speed(vehicle, pix_met, frame_sec, dt=1):

    current = len(vehicle.positions) - 1
    if current >= dt:
        dist = (np.sqrt(np.sum(np.square(np.asarray(vehicle.positions[current]) - np.asarray(vehicle.positions[current - dt]))))) * (1 / pix_met)
        time = dt * (1 / frame_sec)
        speed = (dist / time * 3600) / 1000
        vehicle.current_speed = speed


def lane_detection(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur_image = cv.blur(gray_image, (3, 3))
    cannyed_image = cv.Canny(blur_image, 100, 200)

    lines = cv.HoughLinesP(cannyed_image, rho=6, theta=np.pi / 60, threshold=120, lines=np.array([]), minLineLength=10,
                           maxLineGap=10)
    draw_lines(image, lines, thickness=5)

