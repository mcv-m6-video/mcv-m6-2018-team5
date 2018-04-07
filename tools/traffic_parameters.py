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


def lane_detection(position, lanes):
    for idx in range(len(lanes)):
        polygon = path.Path(np.asarray(lanes[idx]))
        if polygon.contains_point(position):
            return idx
    return -1

