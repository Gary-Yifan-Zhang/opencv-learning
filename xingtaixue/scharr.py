"""Scharr算子和laplacian算子"""
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


scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((scharrxy, laplacian))
cv_show(res, "res")
