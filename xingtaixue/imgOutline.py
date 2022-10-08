"""图像轮廓"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二值化

cir = cv2.imread('circle.jpg')


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
mode:轮廓检索模式
"""
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv_show(thresh, "thresh")

# 二值化
ret_c, thresh_c = cv2.threshold(cir, 127, 255, cv2.THRESH_BINARY)
# cv_show(thresh_c, "thresh_c")

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# 绘制轮廓
# 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
draw_img = img.copy()
res = cv2.drawContours(draw_img, contours, 1, (0, 0, 255), 2)  # -1代表全部特征
cv_show(res, 'res')

# 面积
cnt = contours[1]
print(cv2.contourArea(cnt))
# 周长
print(cv2.arcLength(cnt, True))  # True代表闭合

# 轮廓近似
epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv_show(res, 'res')

# 外接矩形
x, y, w, h = cv2.boundingRect(cnt)
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv_show(img, 'img')

"""
method:轮廓逼近模式
"""
