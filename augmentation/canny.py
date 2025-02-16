import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
image_path = r"D:\python\EGE-UNet\data\tooth_seg_new\train\images\20-front.png"  # 替换为你的牙齿图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 高斯模糊去噪
blurred = cv2.GaussianBlur(image, (5, 5), 0)
# Canny 边缘检测
edges = cv2.Canny(blurred, threshold1=5, threshold2=200)

# # 形态学操作
kernel = np.ones((10, 10), np.uint8)
# # 形态学闭运算
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


# 边缘扩展和填充
#
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for contour in contours:
#     cv2.drawContours(edges, [contour], -1, 255, thickness=2)  # 填充或扩展边缘





# 显示原图和边缘检测结果
plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap="gray")
# plt.axis("off")

plt.subplot(1, 2, 1)
plt.title("Canny Edge Detection")
# plt.imshow(lines , cmap="gray")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Canny Closed Edge Detection")
plt.imshow(closed, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()




#创建边界掩膜
mask = cv2.bitwise_and(image, edges)

# 增强边界区域（提高亮度）
enhanced_edges = cv2.addWeighted(mask, 2, image, 1, 0)

# 将增强后的边界与原图像融合
final_image = cv2.addWeighted(image, 0.7, enhanced_edges, 0.3, 0)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Edge-Enhanced Image")
plt.imshow(final_image, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()



