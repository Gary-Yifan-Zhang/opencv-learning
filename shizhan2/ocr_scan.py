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


def resize(img, width):
    # 等比例改变图像大小
    w, h = img.shape[0], img.shape[1]
    new_w = width
    new_h = int(width * h / w)  # 必须是整数
    print(new_w, new_h)
    new_img = cv2.resize(img, (new_h, new_w))
    return new_img


# 不知道为啥要完整路径，不然报错
img = cv2.imread('D:/pycharmprojects/opencv/shizhan2/picture/receipt.jpg')
cv_show(img, 'img')
# 记录一下原图的比例
ratio = img.shape[0] / 500.0
orig = img.copy()

