"""腐蚀操作"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('opencv.jpg')

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
# cv2.imshow('img', img)
# cv2.imshow('erosion', erosion)
# cv2.waitKey()
# cv2.destroyAllWindows()

cir = cv2.imread('circle.jpg')

kernel = np.ones((30, 30), np.uint8)
erosion_1 = cv2.erode(cir, kernel, iterations=1)
erosion_2 = cv2.erode(cir, kernel, iterations=2)
erosion_3 = cv2.erode(cir, kernel, iterations=3)
res = np.hstack((cir, erosion_1, erosion_2, erosion_3))
cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()
