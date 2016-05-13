import numpy as np

np.seterr(over='ignore')
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def green_channel():
    img = cv2.imread('PeppersBayerGray.bmp', 0)
    IG = img

    # B = (A+C)/2
    # D = (C+H)/2
    # E = (A+I)/2
    # G= (C+F+H+K)/4
    # J= (F+I+K+N)/4
    # L= (H+P)/2
    # M= (I+N)/2
    # O= (N+P)/2

    dup = img
    # goes over mask and interpolates empty entrees
    downshifts = 0

    move_window_vertical = 0
    move_window_vertical2 = 4

    while downshifts < 96:

        rightshifts = 0
        move_window_horizontal = 0
        move_window_horizontal2 = 4
        while rightshifts < 128:
            slice = [dup[i][move_window_horizontal:move_window_horizontal2] for i in
                     range(move_window_vertical, move_window_vertical2)]
            for i in range(4):
                for j in range(4):
                    if i == 0:
                        if j == 1:
                            # left and right
                            slice[i][j] = (slice[i][j - 1] + slice[i][j + 1]) / 2
                        if j == 3:
                            # left and bottom
                            slice[i][j] = (slice[i][j - 1] + slice[i + 1][j]) / 2
                    if i == 1:
                        if j == 0:
                            # top and bottom
                            slice[i][j] = (slice[i - 1][j] + slice[i + 1][j]) / 2
                        if j == 2:
                            # all 4 sides
                            slice[i][j] = (slice[i][j - 1] + slice[i][j + 1] + slice[i - 1][j] + slice[i + 1][j]) / 4
                    if i == 2:
                        if j == 1:
                            slice[i][j] = (slice[i][j - 1] + slice[i][j + 1] + slice[i - 1][j] + slice[i + 1][j]) / 4
                        if j == 3:
                            slice[i][j] = (slice[i - 1][j] + slice[i + 1][j]) / 2
                    if i == 3:
                        if j == 0:
                            # right and top
                            slice[i][j] = (slice[i][j + 1] + slice[i - 1][j]) / 2
                        if j == 2:
                            slice[i][j] = (slice[i][j - 1] + slice[i][j + 1]) / 2
            move_window_horizontal += 4
            move_window_horizontal2 += 4
            rightshifts += 1
        move_window_vertical += 4
        move_window_vertical2 += 4
        downshifts += 1
    return IG
