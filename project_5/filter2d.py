import numpy as np
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# original
img = cv2.imread('ex2.jpg', 0)
cv2.imshow('image', img)
cv2.waitKey(0)

x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
x_kernel = np.asanyarray(x_kernel, np.float32)

y_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
y_kernel = np.asanyarray(y_kernel, np.float32)

x_image = cv2.filter2D(img, -1, x_kernel)
y_image = cv2.filter2D(img, -1, y_kernel)

cv2.imshow('x-image', x_image)
cv2.waitKey(0)
cv2.imshow('y-image', y_image)
cv2.waitKey(0)

result_image = abs(x_image) + abs(y_image)

cv2.imshow("result-image", result_image)
cv2.waitKey(0)
