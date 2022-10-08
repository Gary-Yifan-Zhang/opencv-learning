"""礼帽与黑猫"""
# 礼帽=原始-开运算结果
# 黑帽=闭运算-原始

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 礼帽剩下字
img = cv2.imread('opencv.jpg')
kernel = np.ones((7, 7), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 黑帽剩下刺
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
