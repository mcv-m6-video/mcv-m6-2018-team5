import cv2 as cv
import matplotlib.pyplot as plt

from tools.image_parser import get_image_list_changedetection_dataset

idx = 0


def press(event):
    global idx
    idx += 1
    if idx < len(imageList):
        img = cv.imread(imageList[idx])
        gt = cv.imread(gtList[idx])
        testA_img = cv.imread(testAList[idx]) * 255
        testB_img = cv.imread(testBList[idx]) * 255

        img1.set_data(img)
        img2.set_data(gt)
        img3.set_data(testA_img)
        img4.set_data(testB_img)

        text.set_text('Frame ' + str(idx))

        fig.canvas.draw()


# Get a list with input images filenames
imageList = get_image_list_changedetection_dataset('..\\datasets\\highway\\input', 'in', '001201', 'jpg', 200)

# Get a list with groung truth images filenames
gtList = get_image_list_changedetection_dataset('..\\datasets\\highway\\groundtruth', 'gt', '001201', 'png', 200)

# Get a list with test results filenames
testAList = get_image_list_changedetection_dataset('..\\datasets\\highway\\results_testAB_changedetection',
                                                   str('test_A_'),
                                                   '001201', 'png', 200)
testBList = get_image_list_changedetection_dataset('..\\datasets\\highway\\results_testAB_changedetection',
                                                   str('test_B_'),
                                                   '001201', 'png', 200)

# Display first image
fig, axes = plt.subplots(2, 2, subplot_kw={'xticks': [], 'yticks': []})
fig.canvas.mpl_connect('key_press_event', press)

img = cv.imread(imageList[idx])
gt = cv.imread(gtList[idx])
testA_img = cv.imread(testAList[idx]) * 255
testB_img = cv.imread(testBList[idx]) * 255

img1 = axes.flat[0].imshow(img)
axes.flat[0].set_title('Input image')
img2 = axes.flat[1].imshow(gt)
axes.flat[1].set_title('Ground truth')
img3 = axes.flat[2].imshow(testA_img)
axes.flat[2].set_title('Test A')
img4 = axes.flat[3].imshow(testB_img)
axes.flat[3].set_title('Test B')

text = axes.flat[3].text(.5, .5, 'Frame ' + str(idx), horizontalalignment='right', verticalalignment='bottom')

plt.show()
