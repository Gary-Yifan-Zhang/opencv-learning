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

# 轮廓检测
"""
:param
ref.copy:一定要是复制的图像
cv2.RETR_EXTERNAL:外轮廓
cv2.CHAIN_APPROX_SIMPLE：压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

:returns
refCnts: 轮廓数组
hierarchy：每条轮廓对应的属性
"""
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show(img, 'img')
print(np.array(refCnts).shape) # 打印一下大小，是10个

