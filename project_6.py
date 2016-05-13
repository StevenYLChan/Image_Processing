import numpy as np
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('im1.jpg', 0)
img2 = cv2.imread('im2.jpg', 0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# SIFT algorithm uses Difference of Gaussians which is an approximation of LoG.
# Difference of Gaussian is obtained as the difference of Gaussian blurring of an image

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
# eliminates any low-contrast keypoints and edge keypoints and what remains is strong interest points.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
# If enough matches are found, we extract the locations of matched keypoints in both the images.
# They are passed to find the perpective transformation. Once we get this 3x3 transformation matrix,
# we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it.
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

plt.imshow(img3, 'gray'), plt.show()

# Initialize a matrix to include all the coordinates in the image, from (0, 0), (1, 0), ..., to (w-1, h-1)
c = np.zeros((3, h2 * w2), dtype=numpy.int)
for y in range(h2):
    c[:, y * w2:(y + 1) * w2] = np.matrix([np.arange(w2), [y] * w2, [1] * w2])

# Calculate the new image coordinates.
# M is the homography matrix
new_c = M * np.matrix(c)
new_c = np.around(np.divide(new_c, new_c[2]))

# The new coordinates may have negative values. Perform translation if needed
x_min = np.amin(new_c[0])
y_min = np.amin(new_c[1])
x_max = np.amax(new_c[0])
y_max = np.amax(new_c[1])
if x_min < 0:
    t_x = -x_min
else:
    t_x = 0
if y_min < 0:
    t_y = -y_min
else:
    t_y = 0

# Initialize the final images to include every pixel of the stitched images
new_w = np.maximum(x_max, w1) - np.minimum(x_min, 0) + 1
new_h = np.maximum(y_max, h1) - np.minimum(y_min, 0) + 1
new_img1 = np.zeros((new_h, new_w), dtype=numpy.uint8)
new_img2 = np.zeros((new_h, new_w), dtype=numpy.uint8)

# Assign the first image
new_img1[t_y:t_y + h1, t_x:t_x + w1] = img1

# Assign the second image based on the newly calculated coordinates
for idx in range(c.shape[1]):
    x = c[0, idx]
    y = c[1, idx]
    x_c = new_c[0, idx]
    y_c = new_c[1, idx]
    new_img2[y_c + t_y, x_c + t_x] = img2[y, x]

# The stitched image can be simply obtained by averaging the two final images
new_img = (new_img1 + new_img2) / 2
