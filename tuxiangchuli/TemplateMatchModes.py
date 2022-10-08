"""模板匹配"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('FCB.jpg', 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('Messi.jpg', 0)
h, w = template.shape[:2]


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


res = cv2.matchTemplate(img, template, 1)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # min_loc 最小值所在位置
img = cv2.rectangle(img, min_loc, (min_loc[0] + w, min_loc[1] + h), 255, 0)
cv_show(img, 'img')
