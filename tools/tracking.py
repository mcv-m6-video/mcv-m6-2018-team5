import cv2 as cv
from KalmanFilter import KalmanFilter

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
    boundingBoxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        boundingBoxes.append(BoundingBox(x, y, w, h))

    return boundingBoxes

def kalman_filter_tracking(imgList):

    bbList = extract_bounding_boxes(imgList[0])
    kalman_filter_list = []
    for bb in bbList:
        kalman_filter_list.append(KalmanFilter([bb.cent_x, bb.cent_y]))

    for img in imgList:
        bbList = extract_bounding_boxes(img)
