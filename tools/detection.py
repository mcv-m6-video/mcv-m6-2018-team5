import cv2 as cv
from skimage import io as skio
from skimage import measure as skmeasure
from skimage import morphology as skmorph

def detectObjects(img_data, foreground):
    labeled_image = skmorph.label(foreground, connectivity=foreground.ndim)
    regions_properties = skmeasure.regionprops(labeled_image, img_data)

    bounding_boxes = []
    centroids = []
    for region in regions_properties:
        bbox_x = region.bbox[1]
        bbox_y = region.bbox[0]
        bbox_w = region.bbox[3] - region.bbox[1]
        bbox_h = region.bbox[2] - region.bbox[0]
        bounding_boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])

        centroids.append([region.centroid[1], region.centroid[0]])

    return bounding_boxes, centroids
