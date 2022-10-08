"""傅里叶变换"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('FCB.jpg', 0)


# template = cv2.imread('Messi.jpg', 0)

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 低通滤波器：使图像模糊
# 高通滤波器：使细节增加

# 输入图像先转换成np.float32的格式
# 将频率为0的部分从左上角转换到中间
# cv2.dft()返回的结果为双通道（实部和虚部），需要逆变换成（0，255）
img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# 得到灰度图能表示的格式
# 频域结果
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# 计算图片中心位置
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

print(rows, cols, crow, ccol)

# 低通滤波器
# 将频域图像中心周围一圈（低频部分）保留，进行掩膜操作
mask = np.zeros((rows, cols, 2), np.uint8)  # 双通道（实部和虚部）
mask[crow - 30:crow + 30, ccol - 30: ccol + 30] = 1

# IDFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)  # 再把中间的转换到左上角
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()

# 高通滤波器
# 将频域图像中心周围一圈（低频部分）掩膜掉
mask = np.zeros((rows, cols, 2), np.uint8)  # 双通道（实部和虚部）
mask[crow - 10:crow + 10, ccol - 10: ccol + 10] = 0

# IDFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)  # 再把中间的转换到左上角
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()
