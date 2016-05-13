import numpy as np
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

min = 0
max = 500


def minFunction(new_min):
    global min
    min = new_min
    print('new_min: ', min)
    print('max: ', max)
    new_edges = cv2.Canny(img, new_min, max)
    cv2.imshow('image', new_edges)


def maxFunction(new_max):
    global max
    max = new_max
    print('min: ', min)
    print('new_max: ', max)
    new_edges = cv2.Canny(img, min, new_max)
    cv2.imshow('image', new_edges)


# original
img = cv2.imread('ex2.jpg', 0)

edges = cv2.Canny(img, 0, 0)

cv2.namedWindow('image')
cv2.createTrackbar('min', 'image', min, max, minFunction)
cv2.createTrackbar('max', 'image', min, max, maxFunction)

while (1):
    cv2.imshow('image', img)
    if cv2.waitKey(0):
        break
cv2.destroyAllWindows()
