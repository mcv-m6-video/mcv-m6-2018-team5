import os


# Method to get a list of the image filenames
def get_image_list_changedetection_dataset(dataset_path, image_prefix, first_image_nr, image_type, nr_images):
    image_list = []
    img_nr = int(first_image_nr)
    for i in range(img_nr, img_nr + nr_images):
        tmp_img = image_prefix + str(i).zfill(len(first_image_nr))
        image_path = os.path.join(dataset_path, tmp_img + '.' + image_type)
        image_list.append(image_path)
    return image_list


def get_image_list_kitti_dataset(dataset_path, image_sequences, image_type, image_prefix=''):
    image_list = []
    for img in image_sequences:
        tmp_img = image_prefix + img
        image_path = os.path.join(dataset_path, tmp_img + '.' + image_type)
        image_list.append(image_path)
    return image_list


def get_sequence_list_kitti_dataset(dataset_path, image_sequence, image_type):
    image_list = []

    tmp_img = image_sequence + '_10'
    image_path = os.path.join(dataset_path, tmp_img + '.' + image_type)
    image_list.append(image_path)
    tmp_img = image_sequence + '_11'
    image_path = os.path.join(dataset_path, tmp_img + '.' + image_type)
    image_list.append(image_path)

    return image_list


def get_gt_list_kitti_dataset(gt_path, image_sequence, image_type):
    gtList = []

    tmp_img = image_sequence + '_10'
    image_path = os.path.join(gt_path, tmp_img + '.' + image_type)
    gtList.append(image_path)

    return gtList


def get_image_list_ski_video_dataset(dataset_path, first_image_nr, image_type, nr_images):
    image_list = []
    img_nr = int(first_image_nr)
    for i in range(img_nr, img_nr + nr_images):
        tmp_img = 'frame{}.{}'.format(i, image_type)
        image_path = os.path.join(dataset_path, tmp_img)
        image_list.append(image_path)
    return image_list
