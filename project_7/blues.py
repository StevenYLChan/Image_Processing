import numpy as np

np.seterr(over='ignore')
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def blue_channel():
    img = cv2.imread('PeppersBayerGray.bmp', 0)
    IB = img
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
                    if i == 1:
                        if j == 1:
                            slice[i][j] = (slice[i][j - 1] + slice[i][j + 1]) / 2
                    if i == 2:
                        if j == 0:
                            slice[i][j] = (slice[i - 1][j] + slice[i + 1][j]) / 2
                        if j == 1:
                            slice[i][j] = (slice[i - 1][j - 1] + slice[i - 1][j + 1] + slice[i + 1][j - 1] +
                                           slice[i + 1][j + 1]) / 4
                        if j == 2:
                            slice[i][j] = (slice[i - 1][j] + slice[i + 1][j]) / 2
                    if i == 3:
                        if j == 1:
                            a = slice[i][j - 1]
                            b = slice[i][j + 1]
                            slice[i][j] = (slice[i][j - 1] + slice[i][j + 1]) / 2

            # after we got the values we needed now we can copy it for the empty rows and columns
            for i in range(4):
                for j in range(4):
                    if i == 0:
                        if j == 0:
                            slice[i][j] = slice[i + 1][j]
                        if j == 1:
                            slice[i][j] = slice[i + 1][j]
                        if j == 2:
                            slice[i][j] = slice[i + 1][j]
                    if i == 1:
                        if j == 3:
                            slice[i][j] = slice[i][j - 1]
                    if i == 2:
                        if j == 3:
                            slice[i][j] = slice[i][j - 1]
                    if i == 3:
                        if j == 3:
                            slice[i][j] = slice[i][j - 1]
            # final top right corner
            for i in range(4):
                for j in range(4):
                    if i == 0:
                        if j == 3:
                            slice[i][j] = (slice[i][j - 1] + slice[i + 1][j]) / 2

            move_window_horizontal += 4
            move_window_horizontal2 += 4
            rightshifts += 1
        move_window_vertical += 4
        move_window_vertical2 += 4
        downshifts += 1
    return IB
