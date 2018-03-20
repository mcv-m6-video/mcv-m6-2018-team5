## CODE IMPLEMENTATION FROM: https://github.com/AVINASH793/Video-Stabilization-and-image-mosaicing/blob/master/Video_stabilization.py
                previous_image = image_list[0]
                prev_gray = cv.imread(previous_image, cv.IMREAD_GRAYSCALE)
                transforms = []
                reset_frequency = 100
                previous_to_current_transform = []
                for idx in range(1, len(image_list)):

                    current_image = image_list[idx]
                    curr_gray = cv.imread(current_image, cv.IMREAD_GRAYSCALE)
                    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                    lk_params = dict(winSize=(cf.block_size, cf.block_size), maxLevel=2,
                                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
                    previous_corner = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                    current_corner, st, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, previous_corner, None,
                                                                       **lk_params)

                    previous_corner2 = previous_corner[st == 1]
                    current_corner2 = current_corner[st == 1]

                    previous_corner2, current_corner2 = map(lambda corners: corners[st.ravel().astype(bool)],
                                                  [previous_corner, current_corner])
                    T = cv.estimateRigidTransform(previous_corner2, current_corner2, False)
                    # if(T.any()):
                    #	break
                    if T is not None:
                        dx = T[0, 2]
                        dy = T[1, 2]
                        da = math.atan2(T[1, 0], T[0, 0])
                    else:
                        T = np.eye(3,3)
                        dx = T[0, 2]
                        dy = T[1, 2]
                        da = math.atan2(T[1, 0], T[0, 0])
                        logger.info(idx)

                    previous_to_current_transform.append((dx, dy, da))

                    prev_gray = curr_gray.copy()
                max_frames = len(image_list)-1

                a, x, y = 0.0, 0.0, 0.0
                trajectory = []

                for i in range(len(previous_to_current_transform)):
                    tx, ty, ta = previous_to_current_transform[i]
                    x += tx
                    y += ty
                    a += ta
                    trajectory.append((x, y, a))

                smoothed_trajectory = []

                for i in range(len(trajectory)):
                    sx, sy, sa, ctr = 0.0, 0.0, 0.0, 0
                    for j in range(-SMOOTHING_RADIUS, SMOOTHING_RADIUS + 1):
                        if (i + j >= 0 and i + j < len(trajectory)):
                            tx, ty, ta = trajectory[i + j]
                            sx += tx
                            sy += ty
                            sa += ta
                            ctr += 1
                    smoothed_trajectory.append((sx / ctr, sy / ctr, sa / ctr))

                new_previous_to_current_transform = []
                a, x, y = 0.0, 0.0, 0.0

                for i in range(len(previous_to_current_transform)):
                    tx, ty, ta = previous_to_current_transform[i]
                    sx, sy, sa = smoothed_trajectory[i]
                    x += tx
                    y += ty
                    a += ta
                    new_previous_to_current_transform.append((tx + sx - x, ty + sy - y, ta + sa - a))

                vert_border = HORIZONTAL_BORDER_CROP * len(prev_gray) // len(prev_gray[0])

                for idx in range(0, len(image_list)-1):
                    current_image = image_list[idx]
                    curr = cv.imread(current_image)
                    tx, ty, ta = new_previous_to_current_transform[idx]
                    T = np.matrix([[math.cos(ta), -math.sin(ta), tx], [math.sin(ta), math.cos(ta), ty]])
                    curr2 = cv.warpAffine(curr, T, (len(curr), len(curr[0])))
                    curr2 = curr2[HORIZONTAL_BORDER_CROP:len(curr2[0] - HORIZONTAL_BORDER_CROP),
                            vert_border:len(curr2) - vert_border]
                    rect_image = cv.resize(curr2, (len(curr[0]), len(curr)))
                    if cf.save_results:
                        image_name = os.path.basename(current_image)
                        image_name = os.path.splitext(image_name)[0]
                        cv.imwrite(os.path.join(cf.results_path, image_name + '.' + cf.result_image_type), rect_image)

