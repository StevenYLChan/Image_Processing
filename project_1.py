import numpy as np
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# load image as grayscale using flag 0
img = cv2.imread('test.jpg', 0)

# read image
img = cv2.imread('test.jpg', 1)
# convert RGB to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

# create a 1D array of size 256 with initialized 0's
alist = [0] * 256

rows, columns = img_gray.shape

for i in range(rows):
    for j in range(columns):
        pix_value = img_gray[i][j]
        alist[pix_value] = alist[pix_value] + 1

plt.imshow(img_gray, 'gray')
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
plt.plot(hist)

total_pixels = rows * columns

hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
cumhist = np.cumsum(hist)
eq_list = [0] * 256

newimage = img_gray
rows, columns = img_gray.shape
for i in range(rows):
    for j in range(columns):
        pix_value = img_gray[i][j]
        new_pixval = cumhist[pix_value]
        eq_pix_value = math.floor(((255 / float(total_pixels)) * (new_pixval + 0.5)))
        newimage[i][j] = eq_pix_value

hist = cv2.calcHist([newimage], [0], None, [256], [0, 256])
plt.plot(hist)
img5 = cv2.imread('day.jpg', 0)
hist5 = cv2.calcHist([img5], [0], None, [256], [0, 256])
rows, columns = img5.shape
imgtotalpixels = rows * columns

img2 = cv2.imread('night.jpg', 0)
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
rows, columns = img2.shape
img2totalpixels = rows * columns

hist5 = hist5 / imgtotalpixels
hist2 = hist2 / img2totalpixels

summed = 0;
for i in range(0, 256):
    summed = summed + math.sqrt(hist5[i] * hist2[i])
print summed
