import numpy as np
from skimage.morphology import closing


def hole_filling(image, fourConnectivity = True):
    if fourConnectivity:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    outputImage = closing(image, kernel)
    return outputImage