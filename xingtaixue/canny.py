"""
canny边缘检测
1.使用高斯滤波，过滤噪声
2.计算每个像素点的梯度强度和方向
3.应用非极大值Non—maximum Suppression抑制，以消除边缘检测
4.应用双阈值Double—Threshold检测来确定真实的潜在的边缘
5.通过抑制孤立的弱边缘完成边缘检测
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('D:/pycharmprojects/opencv/xingtaixue/dog.png', cv2.IMREAD_GRAYSCALE)

cir = cv2.imread('D:/pycharmprojects/opencv/xingtaixue/circle.jpg')


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
cv_show(res, 'res')
