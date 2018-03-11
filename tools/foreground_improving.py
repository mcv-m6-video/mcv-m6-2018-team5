import numpy as np
from skimage import morphology

def hole_filling(image, fourConnectivity = True):
    if fourConnectivity:
        outputImage = morphology.remove_small_holes(image, connectivity=1)
    else:
        outputImage = morphology.remove_small_holes(image, connectivity=2)

    return outputImage