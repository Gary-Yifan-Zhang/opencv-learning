"""滤波"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

cat_img = cv2.imread('cat.jpg')
dog_img = cv2.imread('dog.png')

# # 均值滤波
# # 求3x3矩阵9个数的均值
# blur = cv2.blur(cat_img, (5, 5))
# cv2.imshow('org', cat_img)
# cv2.imshow('blur', blur)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 方框滤波
"""
True: 归一化，求均值
False: 不求均值，求和
"""
# box = cv2.boxFilter(cat_img, -1, (2, 2), normalize=False)
# cv2.imshow('org', cat_img)
# cv2.imshow('box', box)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 高斯滤波
"""
卷积核内的权值不一样
权值矩阵
"""
# aussian = cv2.GaussianBlur(cat_img, (5, 5), 1)
# cv2.imshow('org', cat_img)
# cv2.imshow('aussian', aussian)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 中值滤波
"""
中间数
"""
median = cv2.medianBlur(cat_img, 21)
cv2.imshow('org', cat_img)
cv2.imshow('median', median)
cv2.waitKey()
cv2.destroyAllWindows()
