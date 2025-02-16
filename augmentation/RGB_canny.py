import cv2
import numpy as np
import matplotlib.pyplot as plt


def aug_image_with_edge(image, edges):
    # 创建边界掩膜
    mask = cv2.bitwise_and(image, edges)
    # 增强边界区域（提高亮度）
    enhanced_edges = cv2.addWeighted(mask, 20, image, 1, 0)
    # 将增强后的边界与原图像融合
    final_image = cv2.addWeighted(image, 0.7, enhanced_edges, 0.3, 0)
    return final_image

# 读取彩色图像
image = cv2.imread(r"D:\python\EGE-UNet\data\tooth_seg_new\train\images\20-front.png")

# 提取每个颜色通道（BGR顺序）
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]

# 对每个通道应用Canny边缘检测
edges_blue = cv2.Canny(blue_channel, threshold1=100, threshold2=220)
edges_green = cv2.Canny(green_channel, threshold1=90, threshold2=220)
edges_red = cv2.Canny(red_channel, threshold1=180, threshold2=220)

aug_blue = aug_image_with_edge(blue_channel, edges_blue)
aug_green = aug_image_with_edge(green_channel, edges_green)
aug_red = aug_image_with_edge(red_channel, edges_red)


# 可视化每个通道的边缘
plt.figure(figsize=(12, 8))

# 显示蓝色通道边缘
plt.subplot(3, 3, 1)
plt.imshow(edges_blue, cmap='gray')
plt.title('Blue Channel Edges')
plt.axis('off')

# 显示蓝色通道原图
plt.subplot(3, 3, 4)
plt.imshow(blue_channel)
plt.title('Blue Channel')
plt.axis('off')

# 显示增强后的蓝色通道
plt.subplot(3, 3, 7)
plt.imshow(aug_blue)
plt.title('Blue Channel Augmented')
plt.axis('off')

# 显示绿色通道边缘
plt.subplot(3, 3, 2)
plt.imshow(edges_green, cmap='gray')
plt.title('Green Channel Edges')
plt.axis('off')

# 显示绿色通道原图
plt.subplot(3, 3, 5)
plt.imshow(green_channel)
plt.title('Green Channel')
plt.axis('off')

# 显示增强后的绿色通道
plt.subplot(3, 3, 8)
plt.imshow(aug_green)
plt.title('Green Channel Augmented')
plt.axis('off')

# 显示红色通道边缘
plt.subplot(3, 3, 3)
plt.imshow(edges_red, cmap='gray')
plt.title('Red Channel Edges')
plt.axis('off')

# 显示红色通道原图
plt.subplot(3, 3, 6)
plt.imshow(red_channel)
plt.title('Red Channel')
plt.axis('off')

# 显示增强后的红色通道
plt.subplot(3, 3, 9)
plt.imshow(aug_red)
plt.title('Red Channel Augmented')
plt.axis('off')

# 将增强后的通道合并为彩色图像
ori_image = cv2.merge([red_channel, green_channel, blue_channel])
aug_image = cv2.merge([aug_red, aug_green, aug_blue])
# 与原图像进行比较
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(ori_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(aug_image)
plt.title('Augmented Image')
plt.axis('off')

# 显示结果
plt.tight_layout()
plt.savefig("RGB_canny.png", dpi=300, bbox_inches='tight')
plt.show()
