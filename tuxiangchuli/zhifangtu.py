"""图像直方图"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('FCB.jpg', 0)


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# template = cv2.imread('Messi.jpg', 0)

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.hist(img.ravel(), 256)
plt.show()

img = cv2.imread('FCB.jpg', )
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()