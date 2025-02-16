# 使用 Sobel 和 Laplacian 算子进行边缘检测
import cv2
import matplotlib.pyplot as plt
import numpy as np
image_path = r"D:\python\NAFNet-main\demo\tooth.png"  # 替换为你的牙齿图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Sobel 算子 - 计算 x 和 y 方向的梯度
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # x 方向梯度
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # y 方向梯度

# 组合 x 和 y 方向的梯度
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = cv2.convertScaleAbs(sobel_combined)  # 转换为 8-bit 图像

# Laplacian 算子 - 二阶导数边缘检测
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)  # 转换为 8-bit 图像

# 显示原图像和检测结果
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Sobel Combined")
plt.imshow(sobel_combined, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Laplacian")
plt.imshow(laplacian, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
