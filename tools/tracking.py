import cv2 as cv

from kalman_filter import KalmanFilter


class BoundingBox:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cent_x = x + w // 2
        self.cent_y = y + h // 2


def extract_bounding_boxes(img):
    image, contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        bounding_boxes.append(BoundingBox(x, y, w, h))

    return bounding_boxes


def kalman_filter_tracking(img_list):
    bb_list = extract_bounding_boxes(img_list[0])
    kalman_filter_list = []
    for bb in bb_list:
        kalman_filter_list.append(KalmanFilter([bb.cent_x, bb.cent_y]))

    for img in img_list:
        bb_list = extract_bounding_boxes(img)
