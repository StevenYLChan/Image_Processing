import numpy as np
import math
import cv2
import matplotlib

# change backend for osx
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# PART A
import greens, reds, blues

IG = greens.green_channel()
IR = reds.red_channel()
IB = blues.blue_channel()

rgb = np.dstack((IB, IG, IR))
cv2.imshow('rgb', rgb)
cv2.waitKey(0)

# PART B

DR = IR - IG
DB = IB - IG

rows1, columns1 = DR.shape
rows2, columns2 = DB.shape

for i in range(rows1):
    for j in range(columns1):
        if DR[i][j] > 255:
            DR[i][j] == 255
        if DR[i][j] < 0:
            DR[i][j] == 0

for i in range(rows2):
    for j in range(columns2):
        if DB[i][j] > 255:
            DB[i][j] == 255
        if DB[i][j] < 0:
            DB[i][j] == 0

MR = cv2.medianBlur(DR, 3)
MB = cv2.medianBlur(DB, 3)

IRR = MR + IG
IBB = MB + IG

rows3, columns3 = IRR.shape
rows4, columns4 = IBB.shape

for i in range(rows3):
    for j in range(columns3):
        if IRR[i][j] > 255:
            IRR[i][j] == 255
        if IRR[i][j] < 0:
            IRR[i][j] == 0

for i in range(rows4):
    for j in range(columns4):
        if IBB[i][j] > 255:
            IBB[i][j] == 255
        if IBB[i][j] < 0:
            IBB[i][j] == 0

reconstructed = np.dstack((IBB, IG, IRR))
cv2.imshow('reconstructed', reconstructed)
cv2.waitKey(0)
