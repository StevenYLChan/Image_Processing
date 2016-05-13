import numpy as np

np.seterr(over='ignore')
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def circle_detection():
    img = cv2.imread('coins.jpg', 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # input image, detection method, inverse ratio of resolution, minimum distance between detected centers
    # upper threshold for internal canny edge detector, threshold for center detection, min ratio to be detected, max radius to be detected
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=45, param2=30, minRadius=1, maxRadius=50)

    circles = np.uint16(np.around(circles))

    # draw detected circles
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        # cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def thresholding():
    img = cv2.imread('redbloodcell.jpg', 0)

    ret2, th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("th2", th2)
    cv2.waitKey(0)
    cv2.imshow("th3", th3)
    cv2.waitKey(0)
    result = cv2.addWeighted(th2, 0.50, th3, 0.50, 0)

    cv2.imshow('merged', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kmeans_clustering():
    img = cv2.imread('redbloodcell.jpg', 1)
    samples = img.reshape((-1, 3))
    samples = np.float32(samples)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    comp, label, center = cv2.kmeans(samples, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    temp = center[label.flatten()]
    img2 = temp.reshape((img.shape))

    cv2.imshow("orig", img)
    cv2.waitKey(0)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
