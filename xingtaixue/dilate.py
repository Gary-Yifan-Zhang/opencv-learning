"""膨胀操作"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('D:/pycharmprojects/opencv/xingtaixue/opencv.jpg')

kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('img', img)
cv2.imshow('dilate', dilate)
cv2.waitKey()
cv2.destroyAllWindows()

cir = cv2.imread('D:/pycharmprojects/opencv/xingtaixue/circle.jpg')

kernel = np.ones((30, 30), np.uint8)
dilate_1 = cv2.dilate(cir, kernel, iterations=1)
dilate_2 = cv2.dilate(cir, kernel, iterations=2)
dilate_3 = cv2.dilate(cir, kernel, iterations=3)
res = np.hstack((cir, dilate_1, dilate_2, dilate_3))
cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()
