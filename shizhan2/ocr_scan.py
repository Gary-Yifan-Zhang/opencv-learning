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


# 不知道为啥要完整路径，不然报错
img = cv2.imread('D:/pycharmprojects/opencv/shizhan2/picture/receipt.jpg')

cv_show(img, 'img')
