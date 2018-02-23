import os

# Method to get a list of the image filenames
def get_image_list(dataset_path, first_image, image_type, nr_images):
    imageList = []
    #Discard the fist letters and get the image number
    img_str = first_image[2:]
    img_nr = int(img_str)
    for i in range(img_nr, img_nr + nr_images):
        tmp_img = first_image[:2] + str(i).zfill(len(img_str))
        image_path = os.path.join(dataset_path, tmp_img + '.' + image_type)
        imageList.append(image_path)
    return imageList