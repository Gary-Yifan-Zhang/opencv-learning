"""开运算和闭运算"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('opencv.jpg')

kernel = np.ones((5, 5), np.uint8)
# 开：先腐蚀再膨胀
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# 闭：先膨胀再腐蚀
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('img', img)
cv2.imshow('opening', opening)
cv2.imshow('closing', closing)
cv2.waitKey()
cv2.destroyAllWindows()
