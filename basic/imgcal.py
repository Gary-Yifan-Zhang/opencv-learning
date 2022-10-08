"""数值计算"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

cat_img = cv2.imread('cat.jpg')
dog_img = cv2.imread('dog.png')

cat_img2 = cat_img + 30
# cv2.imshow('1', cat_img)
# cv2.imshow('2', cat_img2)
# cv2.imshow('dog', dog_img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(cat_img + cat_img2)  # 对256取余，%
cat_img3 = cv2.add(cat_img, cat_img2)  # 越界取255
# print(cat_img3)
print(dog_img.shape)
cat_img4 = cv2.resize(cat_img, (351, 234))
print(cat_img4.shape)

# cat_img4 = cv2.resize(cat_img, (0, 0), fx=1.5, fy=1.5)  # 倍数
res = cv2.addWeighted(cat_img4, 0.4, dog_img, 0.6, 0)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
