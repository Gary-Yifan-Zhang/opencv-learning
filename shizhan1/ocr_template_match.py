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


def resize(img, width):
    # 等比例改变图像大小
    w, h = img.shape[0], img.shape[1]
    new_w = width
    new_h = int(width * h / w)  # 必须是整数
    print(new_w, new_h)
    new_img = cv2.resize(img, (new_h, new_w))
    return new_img


def digits_sorted(Cnts):
    boundingBoxes = [cv2.boundingRect(c) for c in Cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(Cnts, boundingBoxes),
                                        key=lambda b: b[1][i]))

    return cnts, boundingBoxes
    print("digitCnts_sorted", digitCnts_sorted)


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
print(np.array(refCnts).shape)  # 打印一下大小，是10个

boundingBox = [cv2.boundingRect(c) for c in refCnts]
print(boundingBox)  # （x,y,h,w）

# zip 包装成一个元组 sorted排序，key按照第一个元素排序
(refCnts, boundingBox) = zip(*sorted(zip(refCnts, boundingBox),
                                     key=lambda b: b[1][0], reverse=False))

print(boundingBox)  # 从小到大的排序

# 用字典保存数字的模板
digits = {}
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    # 画一个框
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 灰度处理，改变大小
image = cv2.imread('images/credit_card_01.png')
cv_show(image, 'image')
image = resize(image, width=200)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show(gray, 'gray')

# 礼帽操作，突出明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show(tophat, 'tophat')

sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
# 负数是黑色的
sobelx = cv2.convertScaleAbs(sobelx)  # 绝对值转换

sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)  # 绝对值转换

# 分别计算x和y，再求和
# addweighted求和
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv_show(sobelxy, "sobelxy")

# 归一化一下
sobelxy = np.absolute(sobelxy)
(minVal, maxVal) = (np.min(sobelxy), np.max(sobelxy))
sobelxy = (255 * ((sobelxy - minVal) / (maxVal - minVal)))
sobelxy = sobelxy.astype('uint8')
cv_show(sobelxy, "sobelxy")

# 闭操作，让数字连在一起
sobelxy = cv2.morphologyEx(sobelxy, cv2.MORPH_CLOSE, rectKernel)
cv_show(sobelxy, "sobelxy")

# 再二值化一下 THRESH_OTSU自动寻找合适的阈值
thresh = cv2.threshold(sobelxy, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show(thresh, "thresh")
# 再开操作一下，把细节处理掉，再开操作把空填上
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
cv_show(thresh, "thresh")

# 画出轮廓线
# RETR_EXTERNAL外部轮廓，CHAIN_APPROX_SIMPLE储存简单的信息
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
cnts = contours
# print(cnts)
cur_img = image.copy()
# 画在原始图像之中
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show(cur_img, "cur_img")
locs = []

# 储存轮廓线的位置
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    # 按照长宽比例和长宽大小来筛选合适的轮廓
    ar = w / float(h)
    if ar > 2.4 and ar < 4.2:
        if (w > 40 and w < 56) and (h > 15 and h < 25):
            locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])
print("locs:", locs)
output = []
# 遍历数组，截取对应图像
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    # 稍微都截大一点
    group = gray[gY - 5: gY + gH + 5, gX - 5: gX + gW + 5]
    # print('group:', group.shape)
    # cv_show(group, 'group')

    # 二值化处理一下
    group = cv2.threshold(group, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv_show(group, 'group')
    # 得到轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    cur_group = group.copy()
    # 画出轮廓，但是因为太小了，估计把字遮住了
    cv2.drawContours(cur_group, digitCnts, -1, (0, 0, 255), 3)
    # cv_show(cur_group, "cur_group")
    digitCnts_sorted = digits_sorted(digitCnts)[0]
    for c in digitCnts_sorted:
        print(c)
        (x, y, w, h) = cv2.boundingRect(c)
        # print(x, y, w, h)
        roi = group[y:y + h, x:x + w]
        # cv_show(roi, "roi")
        roi = cv2.resize(roi, (57, 88))
        # cv_show(roi, "roi")

        # 打分
        scores = []
        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
    # 2.3 画出来
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)  # 左上角,右下角
    # 2.4 putText参数：图片,添加的文字,左上角坐标,字体,字体大小,颜色,字体粗细
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 2.5 得到结果
    output.extend(groupOutput)
    print("groupOutput:", groupOutput)

    cv_show(image, 'image')
