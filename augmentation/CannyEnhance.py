import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CannyEnhance(object):
    def __init__(self, low_threshold=100, high_threshold=200, edge_color=(0, 0, 0), alpha=0.5):
        """
        参数:
            low_threshold: Canny 边缘检测的低阈值
            high_threshold: Canny 边缘检测的高阈值
            edge_color: 边缘的颜色（默认为红色），格式为 (R, G, B)
            alpha: 原图与边缘覆盖图的混合比例，范围 [0,1]，alpha 越大边缘效果越明显
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.edge_color = edge_color
        self.alpha = alpha

    def __call__(self, img):
        """
        参数:
            img: PIL Image 格式的 RGB 图像
        返回:
            增强后的 PIL Image，依然为 3 通道 RGB 图像
        """
        # 将 PIL Image 转为 numpy 数组（RGB，数据类型为 uint8）
        img_np = np.array(img)

        # 转换为灰度图，用于 Canny 边缘检测
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 利用 Canny 算法检测边缘，edges 是一个单通道图像，边缘像素值为 255，其余为 0
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

        # 构建一个与原图同尺寸的边缘覆盖图，初始全为 0
        edge_overlay = np.zeros_like(img_np)
        # 将检测到边缘的像素设置为指定的颜色
        edge_overlay[edges != 0] = self.edge_color

        # 混合原图和边缘覆盖图，得到增强效果
        enhanced = cv2.addWeighted(img_np, 1.0, edge_overlay, self.alpha, 0)
        return Image.fromarray(enhanced)

if __name__ == "__main__":
    # 加载测试图像
    img_path = r'D:/python/EGE-UNet/data/tooth_seg_new/train/images/20-front.png'  # 请替换为实际的图片路径
    img = Image.open(img_path).convert('RGB')

    # 定义转换（此处只使用了 CannyEnhance）
    transform = transforms.Compose([
        CannyEnhance(low_threshold=100, high_threshold=200, edge_color=(255, 255, 255), alpha=0.2),
    ])

    # 应用转换，获得增强后的图像
    enhanced_img = transform(img)

    # 可视化对比原图和增强后的图像
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img)
    plt.title("Enhanced Image with Canny")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
# import cv2
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
#
# class CannyEnhanceSoft(object):
#     def __init__(self, low_threshold=100, high_threshold=200, edge_color=(255, 0, 0), alpha=0.5, blur_kernel_size=7):
#         """
#         参数:
#             low_threshold: Canny 边缘检测的低阈值
#             high_threshold: Canny 边缘检测的高阈值
#             edge_color: 边缘的颜色（默认红色），格式为 (R, G, B)
#             alpha: 原图与边缘覆盖图的混合比例，范围 [0, 1]
#             blur_kernel_size: 高斯模糊核的大小（必须为奇数），值越大边缘越柔和
#         """
#         self.low_threshold = low_threshold
#         self.high_threshold = high_threshold
#         self.edge_color = edge_color
#         self.alpha = alpha
#         self.blur_kernel_size = blur_kernel_size
#
#     def __call__(self, img):
#         """
#         参数:
#             img: PIL Image 格式的 RGB 图像
#         返回:
#             增强后的 PIL Image，依然为 3 通道 RGB 图像
#         """
#         # 将 PIL Image 转换为 numpy 数组
#         img_np = np.array(img)
#         # 转换为灰度图，用于 Canny 边缘检测
#         gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#         # 利用 Canny 算法检测边缘
#         edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
#         # 构造一个与原图尺寸相同的边缘覆盖图
#         edge_overlay = np.zeros_like(img_np, dtype=np.uint8)
#         # 将检测到边缘的位置设置为指定的颜色
#         edge_overlay[edges != 0] = self.edge_color
#         # 对边缘覆盖图进行高斯模糊，使描边更加柔和
#         # edge_overlay = cv2.GaussianBlur(edge_overlay, (self.blur_kernel_size, self.blur_kernel_size), 0)
#         # 混合原图和边缘覆盖图，alpha 越大边缘效果越明显
#         enhanced = cv2.addWeighted(img_np, 1.0, edge_overlay, self.alpha, 0)
#         return Image.fromarray(enhanced)
#
# if __name__ == "__main__":
#     # 加载测试图像（请将 'your_image.jpg' 替换为你的图片路径）
#     img_path = 'D:/python/EGE-UNet/data/tooth_seg_new/train/images/20-front.png'
#     img = Image.open(img_path).convert('RGB')
#
#     # 定义转换，此处使用柔化后的 Canny 增强
#     transform = transforms.Compose([
#         CannyEnhanceSoft(low_threshold=100, high_threshold=200, edge_color=(255, 0, 0), alpha=0.7, blur_kernel_size=7),
#     ])
#
#     # 应用转换获得增强后的图像
#     enhanced_img = transform(img)
#
#     # 可视化对比原图和增强后的图像
#     plt.figure(figsize=(12, 6))
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title("Original Image")
#     plt.axis("off")
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(enhanced_img)
#     plt.title("Soft Enhanced Image with Canny")
#     plt.axis("off")
#
#     plt.tight_layout()
#     plt.show()
