"""金字塔"""
"""
高斯金字塔
向下采样方法（缩小）
向上采样方法（放大）
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

cir = cv2.imread('circle.jpg')


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print(img.shape)

up = cv2.pyrUp(img)
cv_show(up, 'up')
print(up.shape)

down = cv2.pyrDown(img)
cv_show(down, "down")
print(down.shape)

"""
拉普拉斯金字塔
L = img - pyrUp(pyrDown(img))
"""
down_up = cv2.pyrUp(down)
down_up = cv2.resize(down_up, (351, 234))
L1 = img - down_up
cv_show(L1, 'L1')
