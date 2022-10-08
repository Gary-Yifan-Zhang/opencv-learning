import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('cat.jpg')
print(img)

cv2.imshow('image', img)

cv2.waitKey(0)
# cv2.destroyWindow()
cv2.destroyAllWindows()


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print(img.shape)  # 三维数值

img2 = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
print(img2)
print(img2.shape)

cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

