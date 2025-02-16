import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch


class AddCannyEdge(object):
    def __init__(self, low_threshold=100, high_threshold=200, resize=(256, 256)):
        self.resize = transforms.Resize(resize)
        self.to_tensor = transforms.ToTensor()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, img):
        # 先进行 resize
        # img = self.resize(img)  # PIL Image

        # 将 PIL Image 转为 numpy 数组
        img_np = np.array(img)  # shape: (H, W, 3)
        if img_np.ndim == 2:  # 如果是灰度图（一般不会发生，因为我们用 .convert('RGB')）
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        # 将 RGB 转为灰度图，用于 Canny 算法
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # 进行 Canny 边缘检测
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)  # 边缘图 shape: (H, W)，值为 0 或 255

        # 将边缘图转换为 PIL Image（模式为 "L"）
        edges_img = Image.fromarray(edges)

        # 将原始 RGB 图像和边缘图分别转换为 tensor，注意各自的 shape 分别为 (3, H, W) 和 (1, H, W)
        rgb_tensor = self.to_tensor(img)
        edge_tensor = self.to_tensor(edges_img)

        # 拼接成四通道 tensor，shape: (4, H, W)
        combined = torch.cat((rgb_tensor, edge_tensor), dim=0)

        # 从Tensor转换为PIL Image
        combined = transforms.ToPILImage()(combined)
        return combined
