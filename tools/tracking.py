import cv2 as cv

def extract_bounding_boxes(img):

    image, contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    bouningBoxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        bouningBoxes.append([x, y, w, h])

    return bouningBoxes