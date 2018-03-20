import numpy as np
import cv2 as cv
import os

def getTrans (videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt, fast=True):
    N_FRAMES = videoArr.shape[0]
    trans = np.zeros((N_FRAMES, 3, 3))

    if fast:
        localMotion = getLocalMotionFast(videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES)
    else:
        localMotion = getLocalMotion(videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES)


    for i in range(N_FRAMES):
        for x in range(3):
            for y in range(3):
                trans[i, x, y] = np.dot(filt, localMotion[i, :, x, y])

    return trans

def getLocalMotion (videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES):
    N_FRAMES = videoArr.shape[0]
    FILT_WIDTH = filt.size
    halfFilt = FILT_WIDTH/2
    localMotion = np.zeros((N_FRAMES, FILT_WIDTH, 3, 3))
    for i in range(N_FRAMES):
        print "frame " + str(i)
        for j in range(FILT_WIDTH):
            if j < halfFilt and i+j-halfFilt >= 0:
                localMotion[i, j, :, :] = np.linalg.inv(localMotion[i+j-halfFilt, FILT_WIDTH-j-1, :, :])
            elif j > halfFilt and i+j-halfFilt <= N_FRAMES-1:
                localMotion[i, j, :, :] = \
                estMotion(videoArr[i, :, :], videoArr[i+j-halfFilt, :, :], detector, bf, MATCH_THRES, RANSAC_THRES, show=False)
            else: # j == halfFilt or out of bound
                localMotion[i, j, :, :] = np.identity(3)
    return localMotion

def getLocalMotionFast (videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES):
    N_FRAMES = videoArr.shape[0]
    FILT_WIDTH = filt.size
    halfFilt = FILT_WIDTH/2
    localMotion = np.zeros((N_FRAMES, FILT_WIDTH, 3, 3))

    # get next frame motion with ORB (and same frame with identity)
    for i in range(N_FRAMES):
        print "frame " + str(i)
        localMotion[i, halfFilt, :, :] = np.identity(3)
        try:
            localMotion[i, halfFilt+1, :, :] = \
            estMotion(videoArr[i, :, :], videoArr[i+1, :, :], detector, bf, MATCH_THRES, RANSAC_THRES, show=False)
        except IndexError:
            localMotion[i, halfFilt+1, :, :] = np.identity(3)

    # get n-step frame motion from next step motion
    for j in range(halfFilt+2, FILT_WIDTH):
        for i in range(N_FRAMES):
            try:
                localMotion[i, j, :, :] = np.dot(localMotion[i+1, j-1, :, :], localMotion[i, j-1, :, :])
            except IndexError:
                localMotion[i, j, :, :] = np.identity(3)

    # get past n-step motion (by inversion of forward motion)
    for j in range(halfFilt):
        for i in range(N_FRAMES):
            try:
                localMotion[i, j, :, :] = np.linalg.inv(localMotion[i+j-halfFilt, FILT_WIDTH-j-1, :, :])
            except IndexError:
                localMotion[i, j, :, :] = np.identity(3)

    return localMotion


def estMotion(frame1, frame2, detector, bf, MATCH_THRES, RANSAC_THRES, show=False):
    try:
        # get keypoints and descriptors
        kp1, des1 = detector.detectAndCompute(frame1, None)
        kp2, des2 = detector.detectAndCompute(frame2, None)

        # get matches
        matches = bf.match(des1, des2)
        matches = filterMatches(matches, MATCH_THRES)

        # get affine transform
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # M, mask = cv.findHomography(src_pts, dst_pts, 0)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, RANSAC_THRES)
        # M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS)

    except:
        M = np.identity(3)

    return M


def filterMatches (matches, MATCH_THRES):
    goodMatches = []
    for m in matches:
        if m.distance < MATCH_THRES:
            goodMatches.append(m)
    return goodMatches


def reconVideo(videoInList, videoOutPath, trans, BORDER_CUT):

    # frame transformation
    for i in range(0, len(videoInList)):
        frame = cv.imread(videoInList[i])
        frameOut = cv.warpPerspective(frame, trans[i, :, :], (frame.shape[0], frame.shape[1], frame.shape[2]), flags=cv.INTER_NEAREST)
        frameOut = frameOut[BORDER_CUT:-BORDER_CUT, BORDER_CUT:-BORDER_CUT]
        image_name = os.path.basename(videoInList[i])
        image_name = os.path.splitext(image_name)[0]
        cv.imwrite(os.path.join(videoOutPath, image_name + '.png'), frameOut)