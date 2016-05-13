import numpy as np
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# original
img = cv2.imread('damaged_cameraman.bmp', 0)
img2 = img
original = img
mask = cv2.imread('damage_mask.bmp', 0)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

damaged_list = []
rows, columns = mask.shape
for i in range(rows):
    for j in range(columns):
        pix_value = mask[i][j]
        if pix_value == 0:
            damaged_list.append([i, j])

print(damaged_list)
rows2, columns2 = blurred.shape
count = 0
while count < 30:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    for i in range(rows2):
        for j in range(columns2):
            if [i, j] in damaged_list:
                img[i][j] = blurred[i][j]
    print(count)
    count = count + 1
    if count == 5:
        cv2.imshow("new", img)
        cv2.waitKey(0)
    if count == 10:
        cv2.imshow("new", img)
        cv2.waitKey(0)
    if count == 20:
        cv2.imshow("new", img)
        cv2.waitKey(0)

cv2.imshow("new", img)
cv2.waitKey(0)
