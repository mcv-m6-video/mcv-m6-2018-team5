import numpy as np
import cv2 as cv
from visualization import draw_lines


def speed_estimation(prevPos, curPos, prevFram, curFram, pix2met, fram2sec):
    dist = (np.sqrt(np.sum(np.square(curPos - prevPos)))) * pix2met
    time = (curFram - prevFram) * fram2sec

    speed = dist / time * (1 / 1000) * 3600

    return speed


def lane_detection(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur_image = cv.blur(gray_image, (3, 3))
    cannyed_image = cv.Canny(blur_image, 100, 200)

    lines = cv.HoughLinesP(cannyed_image, rho=6, theta=np.pi / 60, threshold=120, lines=np.array([]), minLineLength=10,
                           maxLineGap=10)
    draw_lines(image, lines, thickness=5)

