import os
import sys
import cv2
import numpy as np
import math
import time
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# this function reads the ground truth in order to determine intereste points to track
def readTrackingData(filename):
    if not os.path.isfile(filename):
        print "Tracking data file not found:\n ", filename
        sys.exit()

    data_file = open(filename, 'r')
    lines = data_file.readlines()
    no_of_lines = len(lines)
    data_array = np.zeros((no_of_lines, 8))
    line_id = 0
    for line in lines:
        words = line.split()
        if len(words) != 8:
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        coordinates = []
        for word in words:
            coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        line_id += 1
    data_file.close()
    return data_array


# this function writes the corresponding corners to the file
def writeCorners(file_id, corners):
    corner_str = ''
    for i in xrange(4):
        corner_str = corner_str + '{:5.2f}\t{:5.2f}\t'.format(corners[0, i], corners[1, i])
    file_id.write(corner_str + '\n')


# this function draws the bounding box specified by the given corners
def drawRegion(img, corners, color, thickness=1):
    for i in xrange(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)


# this function initializes the tracker with the first frame from the sequence as well as the corresponding corners provided by the ground truth
def initTracker(img, corners):
    global old_gray
    global p0
    old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p0 = corners

    # max corners accuracy/speed trade off
    feature_params = dict(maxCorners=4,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    p10 = p0

    p10[0][0][0] = corners[0][0]
    p10[0][0][1] = corners[1][0]

    p10[1][0][0] = corners[0][1]
    p10[1][0][1] = corners[1][1]

    p10[2][0][0] = corners[0][2]
    p10[2][0][1] = corners[1][2]

    p10[3][0][0] = corners[0][3]
    p10[3][0][1] = corners[1][3]
    p0 = p10


# this function updates the tracker with the current image and returns the corners
def updateTracker(img):
    global old_gray
    global p0

    # termination when done 5 iterations or epsilon better than 0.03. speed vs accuracy trade off
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03))

    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    try:
        corner_array, p0 = convertarraydefault(p0, lk_params, old_gray, frame_gray)
    except:
        corner_array, p0 = convertarray(p0, lk_params, old_gray, frame_gray)

    return corner_array


# this function formats the array
def convertarraydefault(p0, lk_params, old_gray, frame_gray):
    array = np.zeros((2, 4))
    # upperleftX upperrightX lowerrightX lowerleftX
    # upperleftY upperrightY lowerrightY lowerleftY

    array[0][0] = p0[0][0][0]
    array[1][0] = p0[0][0][1]

    array[0][1] = p0[1][0][0]
    array[1][1] = p0[1][0][1]

    array[0][2] = p0[2][0][0]
    array[1][2] = p0[2][0][1]

    array[0][3] = p0[3][0][0]
    array[1][3] = p0[3][0][1]

    return (array, p0)


# this function formats the array
def convertarray(p0, lk_params, old_gray, frame_gray):
    array = np.zeros((2, 4))
    # upperleftX upperrightX lowerrightX lowerleftX
    # upperleftY upperrightY lowerrightY lowerleftY

    array[0][0] = p0[0][0][0]
    array[1][0] = p0[0][0][1]

    array[0][1] = p0[1][0][0]
    array[1][1] = p0[1][0][1]

    array[0][2] = p0[2][0][0]
    array[1][2] = p0[2][0][1]

    feature_params = dict(maxCorners=4,
                          qualityLevel=0.01,
                          minDistance=20,
                          blockSize=7)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    p0 = good_new.reshape(-1, 1, 2)
    array[0][3] = p0[3][0][0]
    array[1][3] = p0[3][0][1]

    return (array, p0)


if __name__ == '__main__':
    sequences = ['bookI', 'bookII', 'bookIII', 'bus', 'cereal']
    seq_id = 1

    if len(sys.argv) > 1:
        seq_id = int(sys.argv[1])

    if seq_id >= len(sequences):
        print 'Invalid dataset_id: ', seq_id
        sys.exit()

    seq_name = sequences[seq_id]
    print 'seq_id: ', seq_id
    print 'seq_name: ', seq_name

    src_fname = seq_name + '/img%03d.jpg'
    ground_truth_fname = seq_name + '.txt'
    result_fname = seq_name + '_res.txt'

    result_file = open(result_fname, 'w')

    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        print 'The video file ', src_fname, ' could not be opened'
        sys.exit()

    # thickness of the bounding box lines drawn on the image
    thickness = 2
    # ground truth location drawn in green
    ground_truth_color = (0, 255, 0)
    # tracker location drawn in red
    result_color = (0, 0, 255)

    # read the ground truth
    ground_truth = readTrackingData(ground_truth_fname)
    no_of_frames = ground_truth.shape[0]

    print 'no_of_frames: ', no_of_frames

    ret, init_img = cap.read()
    if not ret:
        print "Initial frame could not be read"
        sys.exit(0)

    # extract the true corners in the first frame and place them into a 2x4 array
    init_corners = [ground_truth[0, 0:2].tolist(),
                    ground_truth[0, 2:4].tolist(),
                    ground_truth[0, 4:6].tolist(),
                    ground_truth[0, 6:8].tolist()]
    init_corners = np.array(init_corners).T
    # write the initial corners to the result file
    writeCorners(result_file, init_corners)

    # initialize tracker with the first frame and the initial corners
    initTracker(init_img, init_corners)

    # window for displaying the tracking result
    window_name = 'Tracking Result'
    cv2.namedWindow(window_name)

    # lists for accumulating the tracking error and fps for all the frames
    tracking_errors = []
    tracking_fps = []

    for frame_id in xrange(1, no_of_frames):
        ret, src_img = cap.read()
        if not ret:
            print "Frame ", frame_id, " could not be read"
            break
        actual_corners = [ground_truth[frame_id, 0:2].tolist(),
                          ground_truth[frame_id, 2:4].tolist(),
                          ground_truth[frame_id, 4:6].tolist(),
                          ground_truth[frame_id, 6:8].tolist()]
        actual_corners = np.array(actual_corners).T

        start_time = time.clock()
        # update the tracker with the current frame
        tracker_corners = updateTracker(src_img)
        end_time = time.clock()

        # write the current tracker location to the result text file
        writeCorners(result_file, tracker_corners)

        # compute the tracking fps
        current_fps = 1.0 / (end_time - start_time)
        tracking_fps.append(current_fps)

        # compute the tracking error
        current_error = math.sqrt(np.sum(np.square(actual_corners - tracker_corners)) / 4)
        tracking_errors.append(current_error)

        # draw the ground truth location
        drawRegion(src_img, actual_corners, ground_truth_color, thickness)
        # draw the tracker location
        drawRegion(src_img, tracker_corners, result_color, thickness)
        # write statistics (error and fps) to the image
        cv2.putText(src_img, "{:5.2f} {:5.2f}".format(current_fps, current_error), (5, 15),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
        # display the image
        cv2.imshow(window_name, src_img)

        if cv2.waitKey(1) == 27:
            break
            # print 'curr_error: ', curr_error

    mean_error = np.mean(tracking_errors)
    mean_fps = np.mean(tracking_fps)

    print 'mean_error: ', mean_error
    print 'mean_fps: ', mean_fps

    result_file.close()
