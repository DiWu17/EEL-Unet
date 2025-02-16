import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image = cv2.imread('image.jpg')  # 请替换为你的文件路径

# 2. 转换到 HSV 颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 提取色调通道（H）
h_channel = hsv_image[:, :, 0]

# 4. 计算梯度（使用 Sobel 算子）
# 计算水平和垂直方向的梯度
grad_x = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
grad_y = cv2.Sobel(h_channel, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向

# 计算梯度的幅度
magnitude = cv2.magnitude(grad_x, grad_y)

# 5. 可视化结果
plt.figure(figsize=(12, 8))

# 显示原始色调通道
plt.subplot(2, 3, 1)
plt.imshow(h_channel, cmap='gray')
plt.title('Hue Channel')
plt.axis('off')

# 显示水平梯度
plt.subplot(2, 3, 2)
plt.imshow(grad_x, cmap='gray')
plt.title('Gradient in X')
plt.axis('off')

# 显示垂直梯度
plt.subplot(2, 3, 3)
plt.imshow(grad_y, cmap='gray')
plt.title('Gradient in Y')
plt.axis('off')

# 显示梯度幅度（颜色梯度）
plt.subplot(2, 3, 4)
plt.imshow(magnitude, cmap='hot')
plt.title('Gradient Magnitude')
plt.axis('off')

# 显示原始图像
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 显示结果
plt.tight_layout()
plt.show()
