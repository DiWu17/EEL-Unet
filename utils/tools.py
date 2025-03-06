import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math

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




def gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    """
    构造高斯核，返回形状为 (channels, 1, kernel_size, kernel_size) 的核。
    """
    # 构建坐标轴（中心在 (kernel_size-1)/2）
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur_torch(img, kernel_size=5, sigma=1.0):
    """
    对输入图像 img 进行高斯平滑。
    img: (N, C, H, W)
    """
    channels = img.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(img.device).type(img.dtype)
    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel, padding=padding, groups=channels)
    return blurred


def sobel_filters():
    """
    返回 Sobel 算子的 x、y 核，形状均为 (1,1,3,3)
    """
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], dtype=torch.float32)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    return sobel_x, sobel_y


def non_max_suppression_vectorized(grad_magnitude, grad_angle):
    """
    矢量化实现非极大值抑制。

    参数:
      grad_magnitude: (N, 1, H, W) 梯度幅值
      grad_angle: (N, 1, H, W) 梯度角度（单位为度，范围 [0, 180)）

    返回:
      suppressed: (N, 1, H, W) 非极大值抑制后的结果
    """
    # 先将角度归一到 [0,180)
    angle = grad_angle % 180
    # 将角度量化为 0°, 45°, 90°, 135° 四个方向
    q = torch.zeros_like(angle)
    q[(angle < 22.5) | (angle >= 157.5)] = 0
    q[(angle >= 22.5) & (angle < 67.5)] = 45
    q[(angle >= 67.5) & (angle < 112.5)] = 90
    q[(angle >= 112.5) & (angle < 157.5)] = 135

    mag = grad_magnitude

    # 利用 F.pad 构造出相邻像素（方向不同）：
    mag_left = F.pad(mag, (1, 0, 0, 0))[:, :, :, :-1]
    mag_right = F.pad(mag, (0, 1, 0, 0))[:, :, :, 1:]
    mag_up = F.pad(mag, (0, 0, 1, 0))[:, :, :-1, :]
    mag_down = F.pad(mag, (0, 0, 0, 1))[:, :, 1:, :]
    mag_up_left = F.pad(mag, (1, 0, 1, 0))[:, :, :-1, :-1]
    mag_up_right = F.pad(mag, (0, 1, 1, 0))[:, :, :-1, 1:]
    mag_down_left = F.pad(mag, (1, 0, 0, 1))[:, :, 1:, :-1]
    mag_down_right = F.pad(mag, (0, 1, 0, 1))[:, :, 1:, 1:]

    # 对于不同方向的像素，比较当前像素与相邻两个像素的大小：
    mask0 = (q == 0)
    mask45 = (q == 45)
    mask90 = (q == 90)
    mask135 = (q == 135)

    cond0 = (mag >= mag_left) & (mag >= mag_right)
    cond45 = (mag >= mag_up_right) & (mag >= mag_down_left)
    cond90 = (mag >= mag_up) & (mag >= mag_down)
    cond135 = (mag >= mag_up_left) & (mag >= mag_down_right)

    cond = (mask0 & cond0) | (mask45 & cond45) | (mask90 & cond90) | (mask135 & cond135)
    suppressed = mag * cond.float()
    return suppressed


def canny_edge_torch(binary_mask, low_threshold=0.2, high_threshold=0.8,
                     gaussian_kernel_size=5, gaussian_sigma=1.0):
    """
    改进版 Canny 边缘检测算法，适配 (N,1,H,W) 输入。

    参数:
      binary_mask: 输入图像，(N,1,H,W)，假定值在 [0,1] 范围内
      low_threshold, high_threshold: 双阈值（可调参数）
      gaussian_kernel_size, gaussian_sigma: 高斯平滑参数
      gaussian_sigma: 高斯核标准差
    返回:
      edges: 二值化边缘图，(N,1,H,W)，边缘像素值为 1，其余为 0
    """
    # 1. 高斯平滑
    blurred = gaussian_blur_torch(binary_mask, kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)

    # 2. 计算梯度（使用 Sobel 算子）
    sobel_x, sobel_y = sobel_filters()
    sobel_x = sobel_x.to(binary_mask.device).type(binary_mask.dtype)
    sobel_y = sobel_y.to(binary_mask.device).type(binary_mask.dtype)
    grad_x = F.conv2d(blurred, sobel_x, padding=1)
    grad_y = F.conv2d(blurred, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_angle = torch.atan2(grad_y, grad_x) * 180 / math.pi  # 转为角度
    grad_angle[grad_angle < 0] += 180  # 保证在 [0,180)

    # 3. 非极大值抑制
    nms = non_max_suppression_vectorized(grad_magnitude, grad_angle)

    # 4. 双阈值处理
    # 将大于 high_threshold 的定为强边缘，小于 low_threshold 的舍去，
    # 介于两者之间的暂定为弱边缘（此处简单处理，直接保留弱边缘）
    strong = (nms >= high_threshold).float()
    weak = ((nms >= low_threshold) & (nms < high_threshold)).float()
    edges = strong + weak
    edges[edges > 0] = 1.0  # 二值化
    return edges


def canny_edge_torch_improve(binary_mask, low_threshold=0.2, high_threshold=0.7,
                             gaussian_kernel_size=5, gaussian_sigma=1.0):
    """
    改进版 Canny 边缘检测算法，适配 (N,1,H,W) 输入。

    参数:
      binary_mask: 输入图像，(N,1,H,W)，假定值在 [0,1] 范围内
      low_threshold, high_threshold: 双阈值（可调参数）
      gaussian_kernel_size, gaussian_sigma: 高斯平滑参数
    返回:
      edges: 二值化边缘图，(N,1,H,W)，边缘像素值为 1，其余为 0
    """
    # 1. 高斯平滑
    blurred = gaussian_blur_torch(binary_mask, kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)

    # 2. 计算梯度（使用 Sobel 算子）
    sobel_x, sobel_y = sobel_filters()
    sobel_x = sobel_x.to(binary_mask.device).type(binary_mask.dtype)
    sobel_y = sobel_y.to(binary_mask.device).type(binary_mask.dtype)

    # 反射填充，减少边界伪影
    blurred_padded = F.pad(blurred, (1, 1, 1, 1), mode="reflect")

    grad_x = F.conv2d(blurred_padded, sobel_x)
    grad_y = F.conv2d(blurred_padded, sobel_y)

    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    epsilon = 1e-6
    grad_angle = torch.atan2(grad_y, grad_x + epsilon) * 180 / math.pi  # 角度计算稳定
    grad_angle[grad_angle < 0] += 180  # 保证在 [0,180)

    # 3. 非极大值抑制 (优化 nms 计算方式)
    nms = non_max_suppression_vectorized(grad_magnitude, grad_angle)

    # 4. 双阈值处理
    strong = (nms >= high_threshold).float()
    weak = ((nms >= low_threshold) & (nms < high_threshold)).float()

    # 5. 采用 hysteresis 连接弱边缘
    edges = hysteresis_thresholding(strong, weak)

    return edges


def hysteresis_thresholding(strong, weak):
    """
    连接弱边缘到强边缘
    """
    kernel = torch.tensor([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=torch.float32, device=strong.device).unsqueeze(0).unsqueeze(0)

    strong = strong.bool()  # 转换为 bool 类型
    weak = weak.bool()

    while True:
        strong_new = F.conv2d(strong.float(), kernel, padding=1) > 0  # 计算连通性，并转换回 bool
        if torch.equal(strong_new, strong):  # 直到不再变化
            break
        strong = strong_new | weak  # 使用 bool 类型的 | 操作

    return strong.float()  # 转回 float 以匹配原始数据类型

