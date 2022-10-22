__author_ = 'Gary Zhang'

import cv2
from imutils import contours
import imutils
import numpy as np
import argparse
from scipy.spatial import distance as dist


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def transform(img, screenCnt):
    ord = order_points(screenCnt)
    (tl, tr, br, bl) = ord

    # 求边长
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    heightB = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后的坐标
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32'
    )

    M = cv2.getPerspectiveTransform(ord, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


# 不知道为啥要完整路径，不然报错
img = cv2.imread('D:/pycharmprojects/opencv/shizhan2/picture/receipt.jpg')
# 这也太大了
cv_show(img, 'img')
# 记录一下原图的比例
ratio = img.shape[0] / 500.0
orig = img.copy()

img = resize(orig, height=500)
cv_show(img, 'img')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
cv_show(gray, 'gray')
cv_show(edged, 'edged')

# 找到文件的外轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)[0]

# img_cnts = img.copy()
# cv2.drawContours(img_cnts, cnts, -1, (0, 0, 255), 3)
# cv_show(img_cnts, 'img_cnts')

# 外轮廓一定是最大的几个，直接排序操作
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in cnts:
    peri = cv2.arcLength(c, True)
    # 0.02 * length 精度控制
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    print(len(approx))
    # 四个点的就拿出来（四个边的）
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
cv_show(img, 'img')

warped = transform(orig, screenCnt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 80, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)
cv_show(ref, 'ref')
cv_show(resize(ref, height=650), 'ref')
