import cv2 as cv
from skimage import io as skio
from skimage import measure as skmeasure
from skimage import morphology as skmorph

def detectObjects(image_path, foreground):
    labeled_image = skmorph.label(foreground, connectivity=foreground.ndim)
    img_data = skio.imread(image_path, as_grey=True)
    regions_properties = skmeasure.regionprops(labeled_image, img_data)

    bounding_boxes = [region.bbox for region in regions_properties]
    centroids = [region.centroid for region in regions_properties]

    return bounding_boxes, centroids
