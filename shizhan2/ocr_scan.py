__author_ = 'Gary Zhang'

import cv2
from imutils import contours
import imutils
import numpy as np
import argparse


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(img, hight):
    # 等比例改变图像大小
    w, h = img.shape[0], img.shape[1]
    new_h = hight
    new_w = int(hight * w / h)  # 必须是整数
    print("新的长和宽：", new_w, new_h)
    new_img = cv2.resize(img, (new_h, new_w))
    return new_img


# 不知道为啥要完整路径，不然报错
img = cv2.imread('D:/pycharmprojects/opencv/shizhan2/picture/receipt.jpg')
# 这也太大了
cv_show(img, 'img')
# 记录一下原图的比例
ratio = img.shape[0] / 500.0
orig = img.copy()

img = resize(orig, hight=500)
cv_show(img, 'img')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
cv_show(gray, 'gray')
cv_show(edged, 'edged')

# 找到文件的外轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)[0]

# cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
# cv_show(img, 'img')

# 外轮廓一定是最大的几个，直接排序操作
cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in cnts:
    length = cv2.arcLength(c, True)
    # 0.02 * length精度控制
    approx = cv2.approxPolyDP(c, 0.02 * length, True)
    if len(approx) == 4:
        screencnt = approx
        break