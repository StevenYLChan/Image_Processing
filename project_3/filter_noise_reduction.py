import numpy as np
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# original
img = cv2.imread('noisy.jpg')
cv2.imshow("original", img)
cv2.waitKey(0)

# median filter
# The median filter run through each element of the signal
# (in this case the image) and replace each pixel with the median of its neighboring pixels
# (located in a square neighborhood around the evaluated pixel).
median = cv2.medianBlur(img, 5)
cv2.imshow("median filter", median)
cv2.waitKey(0)

# gaussian filter
# src,gausian kernel size 5x5 dimensions. standard deviation in x,y direction.
# since 0 they are computed from ksize.width and ksize.height , respectively
blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("gaussian", blur)
cv2.waitKey(0)
