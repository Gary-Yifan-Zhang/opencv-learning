"""计算梯度"""
# 边缘检测
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('dog.png',cv2.IMREAD_GRAYSCALE)

cir = cv2.imread('circle.jpg')


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sobelx = cv2.Sobel(cir, cv2.CV_64F, 1, 0, ksize=3)
# 负数是黑色的
sobelx = cv2.convertScaleAbs(sobelx)  # 绝对值转换
cv_show(sobelx, 'sobelx')

sobely = cv2.Sobel(cir, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)  # 绝对值转换
cv_show(sobely, 'sobely')

# 分别计算x和y，再求和
# addweighted求和
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show(sobelxy, "sobelxy")

# 不建议直接求和,效果一般
sobelxy = cv2.Sobel(cir, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)  # 绝对值转换
cv_show(sobelxy, 'sobelxy')

cat_sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
cat_sobelx = cv2.convertScaleAbs(cat_sobelx)  # 绝对值转换
cat_sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
cat_sobely = cv2.convertScaleAbs(cat_sobely)  # 绝对值转换
cat_sobelxy = cv2.addWeighted(cat_sobelx, 0.5, cat_sobely, 0.5, 0)
cv_show(cat_sobelxy, "cat_sobelxy")
