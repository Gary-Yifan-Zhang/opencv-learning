import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('cat.jpg')

b, g, r = cv2.split(img)
print(b)
print(b.shape)

# img = cv2.merge(b, g, r)
print(img.shape)
cur_img = img.copy()
# cur_img[:, :, 0] = 0
cur_img[:, :, 1] = 0
cur_img[:, :, 2] = 0
cv2.imshow('R', cur_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
