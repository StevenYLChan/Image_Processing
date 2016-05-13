import numpy as np
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

img = cv2.imread('audi.jpg', 0)
plt.imshow(img)

# kernel is coefficients of filter
kernel = -np.ones((3, 3), np.float32)
kernel[1][1] = 8;
dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
