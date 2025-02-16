import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(r"D:\tempdataset\tooth_new\tooth_seg_new\images\11-front.png")

# 转换到 HSV 颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 提取色调通道
h_channel = hsv_image[:, :, 2]
edges_green = cv2.Canny(h_channel, threshold1=180, threshold2=220)

# 使用 Matplotlib 显示色调通道
plt.imshow(edges_green, cmap='gray')
plt.title('Hue Channel')
plt.axis('off')  # 关闭坐标轴
plt.show()
