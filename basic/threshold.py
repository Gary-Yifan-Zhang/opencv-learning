"""阈值"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

cat_img = cv2.imread('cat.jpg')
dog_img = cv2.imread('dog.png')
gray = cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY)  # 灰度图
print(gray.shape)
ret, thresh1 = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)  # 127-255=255 0-127=0 二值化
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # binary的翻转
ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)  # 0-127不变 大于阈值的等于阈值
ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)  # 0-127变成0，其他不变
ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)  # tozero的inverse

titles = ['Original Image', 'BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
imgs = [cat_img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(imgs[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
# cv2.imshow('1', thresh1)
# cv2.waitKey()
# cv2.destroyAllWindows()
