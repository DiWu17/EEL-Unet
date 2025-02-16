import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def visualize_images(arr, title="8张图片"):
    """
    可视化形状为 (8, 1, 256, 256) 的 numpy 数组，每一张图片为灰度图像，并为整个图设置一个标题。

    参数:
        arr: numpy.ndarray, 形状为 (8, 1, 256, 256)
        title: str, 整个图的标题（默认为 "8张图片"）
    """
    # 检查输入数组是否满足要求
    if arr.ndim != 4 or arr.shape[0] != 8 or arr.shape[1] != 1:
        print("输入数组形状应为 (8, 1, 256, 256)")
        print(f"当前输入数组形状为 {arr.shape}")
        return
        # raise ValueError("输入数组形状应为 (8, 1, 256, 256)")

    # 创建2行4列的子图
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # 设置整个图的标题
    fig.suptitle(title, fontsize=16)

    # 遍历8张图片，取第2个维度索引0作为通道
    for i, ax in enumerate(axes.flat):
        img = arr[i, 0, :, :]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Image {i + 1}")
        ax.axis('off')

    # 调整布局，使标题不会遮挡子图
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.imsave("results/edge_gt.png", arr[0, 0, :, :], cmap='gray')
    plt.show()

def generate_edge_label(gt):
    """
    gt: numpy数组格式的单通道分割标签，形状为 (N, 1, H, W)，数据类型可以不是 uint8
    返回：一个归一化到 [0,1] 的二值边缘图，形状为 (N, 1, H, W)
    """
    # 确保输入是 numpy 数组
    gt = np.array(gt)

    N, C, H, W = gt.shape
    if C != 1:
        raise ValueError("输入数组的通道数必须为1")

    edge_list = []
    # 遍历 batch 中的每一幅图像
    for i in range(N):
        # visualize_image_np(gt[i, 0], "Ground Truth Images")
        # 去除通道维度，得到 (H, W) 的图像
        single_img = (gt[i, 0]*255).astype(np.uint8)
        # 使用 Canny 算子提取边缘
        edges = cv2.Canny(single_img, threshold1=100, threshold2=200)
        # visualize_image_np(edges, "Edge Images")
        # 将边缘图归一化到 [0, 1]，转换为 float32
        edges = edges.astype(np.float32) / 255.0
        # 增加通道维度，恢复为 (1, H, W)
        edges = np.expand_dims(edges, axis=0)
        # visualize_image_np(edges, "Edge Images")
        edge_list.append(edges)
    # 将所有结果堆叠，得到 (N, 1, H, W)
    edge_labels = np.stack(edge_list, axis=0)
    return torch.from_numpy(edge_labels).float()


def visualize_image_np(image, title=None):
    """
    可视化 numpy 数组图像。
    参数:
        image: numpy.ndarray，可能具有以下形状之一：
               - [H, W] (灰度图)
               - [1, H, W] (灰度图，通道在最前)
               - [3, H, W] (RGB图像，通道在最前)
               - [H, W, 1] (灰度图，通道在最后)
               - [H, W, 3] (RGB图像，通道在最后)
        title: 可选的图像标题
    """
    # 如果是单通道二维数组，直接显示为灰度图
    if image.ndim == 2:
        img = image
        cmap = 'gray'
    elif image.ndim == 3:
        # 判断通道维度的位置
        if image.shape[0] in [1, 3]:
            # 通道在最前：形状为 [C, H, W]
            if image.shape[0] == 1:
                # 灰度图
                img = image.squeeze(0)
                cmap = 'gray'
            else:
                # RGB 图像，转换为 [H, W, C]
                img = np.transpose(image, (1, 2, 0))
                cmap = None
        elif image.shape[2] in [1, 3]:
            # 通道在最后：形状为 [H, W, C]
            if image.shape[2] == 1:
                img = image.squeeze(2)
                cmap = 'gray'
            else:
                img = image
                cmap = None
        else:
            raise ValueError("Unsupported image shape: expected channel dimension to be 1 or 3.")
    else:
        raise ValueError("Unsupported image dimension: expected 2 or 3 dimensions.")

    # 如果图像为浮点型且数值范围超过1，则归一化到 [0,1]
    if np.issubdtype(img.dtype, np.floating):
        if img.max() > 1:
            img = img / 255.0

    plt.figure()
    plt.imshow(img, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()


def rgb_to_grayscale(batch):
    """
    将输入的三通道图像 tensor 转换为灰度图。

    参数:
      batch: torch.Tensor，尺寸为 (N, 3, H, W)，其中 N 是批大小
    返回:
      gray: torch.Tensor，尺寸为 (N, 1, H, W)，灰度图像
    """
    # 分别提取 R, G, B 通道（注意这里保持通道维度不变）
    r = batch[:, 0:1, :, :]
    g = batch[:, 1:2, :, :]
    b = batch[:, 2:3, :, :]

    # 按照加权公式转换为灰度图
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray
