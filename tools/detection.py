import cv2 as cv


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