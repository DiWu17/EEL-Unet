import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils import generate_edge_label


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






def extract_edges_torch(binary_mask):
    """
    从二值化 mask 中提取边缘。

    输入:
        binary_mask: torch.Tensor，形状为 (N, 1, H, W)，
                     假设前景像素为 1，背景像素为 0
    输出:
        edges: torch.Tensor，形状与 binary_mask 相同，
               边缘像素为 1，其它像素为 0
    """
    # 定义 3x3 全 1 的卷积核，与输入 tensor 同 device 和 dtype
    kernel = torch.ones((1, 1, 3, 3), dtype=binary_mask.dtype, device=binary_mask.device)

    # 通过卷积计算每个像素及其 3x3 邻域内所有值的和
    # 使用 padding=1 保持输出尺寸与输入一致（边缘处自动补 0）
    neighbor_sum = F.conv2d(binary_mask, kernel, padding=1)

    # 对于每个前景像素（==1），如果其邻域（包括自身）的和小于 9，则说明至少存在一个背景邻域，
    # 该像素被视为边缘像素
    edges = ((binary_mask == 1) & (neighbor_sum < 9)).to(binary_mask.dtype)

    return edges

class EdgeUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeUnet, self).__init__()

        # 模型名称
        self.name = "edgeunet"

        # 编码器部分
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 瓶颈层
        self.bottleneck = self.conv_block(512, 1024)

        # 解码器部分
        self.upconv4 = self.upconv_block(1024, 512)
        self.conv4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.conv3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.conv2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.conv1 = self.conv_block(128, 64)

        # 最终输出层（语义分割结果）
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # 辅助边缘分支：利用最后一个解码器特征生成1通道边缘预测图
        self.edge_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        # 定义卷积块，包括两个卷积层和ReLU激活函数
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        # 定义上采样块，使用反卷积层
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def center_crop(self, layer, target_size):
        """对特征图进行中心裁剪，使其与目标大小匹配"""
        _, _, h, w = layer.size()
        target_h, target_w = target_size[2], target_size[3]
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return layer[:, :, top:top + target_h, left:left + target_w]

    def forward(self, x):
        # 编码器部分
        enc1 = self.enc1(x)
        enc2 = nn.MaxPool2d(kernel_size=2)(enc1)
        enc2 = self.enc2(enc2)
        enc3 = nn.MaxPool2d(kernel_size=2)(enc2)
        enc3 = self.enc3(enc3)
        enc4 = nn.MaxPool2d(kernel_size=2)(enc3)
        enc4 = self.enc4(enc4)
        bottleneck = nn.MaxPool2d(kernel_size=2)(enc4)
        bottleneck = self.bottleneck(bottleneck)

        # 解码器部分
        dec4 = self.upconv4(bottleneck)
        enc4_crop = self.center_crop(enc4, dec4.size())
        dec4 = torch.cat((dec4, enc4_crop), dim=1)  # 跳跃连接
        dec4 = self.conv4(dec4)

        dec3 = self.upconv3(dec4)
        enc3_crop = self.center_crop(enc3, dec3.size())
        dec3 = torch.cat((dec3, enc3_crop), dim=1)  # 跳跃连接
        dec3 = self.conv3(dec3)

        dec2 = self.upconv2(dec3)
        enc2_crop = self.center_crop(enc2, dec2.size())
        dec2 = torch.cat((dec2, enc2_crop), dim=1)  # 跳跃连接
        dec2 = self.conv2(dec2)

        dec1 = self.upconv1(dec2)
        enc1_crop = self.center_crop(enc1, dec1.size())
        dec1 = torch.cat((dec1, enc1_crop), dim=1)  # 跳跃连接
        dec1 = self.conv1(dec1)

        # 主分支输出：语义分割结果
        seg_out = self.final_conv(dec1)

        # 辅助分支输出：边缘预测
        edge_out = self.edge_conv(dec1)
        # 为了便于监督，通常对边缘分支输出使用 Sigmoid 得到 [0,1] 概率
        edge_out = torch.sigmoid(edge_out)

        # print(edge_out)

        binary_edge_out = (edge_out > 0.5).float()
        # print(binary_edge_out)
        # print(binary_edge_out.s)

        binary_edge_out = canny_edge_torch(binary_edge_out)
        # binary_edge_out = canny_edge_torch_improve(binary_edge_out)
        # binary_edge_out = generate_edge_label(binary_edge_out.cpu().numpy())

        # 返回两个输出，训练时分别计算损失
        return seg_out, binary_edge_out

