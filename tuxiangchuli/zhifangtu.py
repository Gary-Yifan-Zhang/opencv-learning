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


hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 灰度图
plt.hist(img.ravel(), 256)
plt.show()

img = cv2.imread('FCB.jpg')
color = ('b', 'g', 'r')  # rgb图
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

# 创建mask
# 掩膜就是一个黑框，变成255，拿出图片需要的部分
mask = np.zeros(img.shape[0:2], np.uint8)
print(mask.shape)
mask[50:250, 100:350] = 255  # 将一定范围的像素转换为255
cv_show(mask, "mask")

img = cv2.imread('FCB.jpg', 0)
cv_show(img, "img")  # 显示原始灰度图像
mask_img = cv2.bitwise_and(img, img, mask=mask)  # 对图像的像素进行与操作
cv_show(mask_img, 'mask_img')

# 均衡处理
# 让直方图更均衡，图像更清晰，边界点更明显
# 映射操作：从一个分布映射到另外一个操作 累计概率*取值范围 再取整
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(), 256)
plt.show()

res = np.hstack((img, equ))
cv_show(res, 'res')

# 可能会丢失细节，因为对整体图像做了平均
# 自适应直方图均衡化
# 对部分分别做均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clahe.apply(img)
res = np.hstack((img, equ, res_clahe))
cv_show(res, 'res')
