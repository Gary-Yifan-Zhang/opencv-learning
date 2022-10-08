"""梯度操作"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('opencv.jpg')
cir = cv2.imread('circle.jpg')
# 梯度=膨胀-腐蚀
# 得到轮廓
kernel = np.ones((5, 5), np.uint8)
gradient = cv2.morphologyEx(cir, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
