"""识别银行卡上的数字"""

# 基本方法：模板匹配
# 1. 取出外轮廓
# 2. 把数字拿出了：外接矩形
# 3. 模板匹配

import cv2
from imutils import contours
import imutils
import numpy as np
import argparse


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('images/numbers.png')
cv_show(img, 'img')
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(ref, 'ref')
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]  # [1]：返回第二个返回值
cv_show(ref, 'ref')
