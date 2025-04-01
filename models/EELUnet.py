import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils.tools import canny_edge_torch, visualize_images, calculate_contribution


class EdgeGuidedUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(EdgeGuidedUpsample, self).__init__()

        # 参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor

        # 边缘提取模块 (使用简单的 3x3 卷积模拟边缘检测)
        self.edge_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,  # 单通道边缘图
            kernel_size=3,
            padding=1,
            bias=False
        )
        # 初始化为类似 Sobel 的边缘检测核
        with torch.no_grad():
            self.edge_conv.weight.data = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, in_channels, 1, 1)

        # 上采样模块 (使用转置卷积)
        self.upconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=upscale_factor,
            padding=1,
            output_padding=0
        )

        # 边缘引导融合层 (将边缘图与上采样结果融合)
        self.fusion_conv = nn.Conv2d(
            in_channels=out_channels + 1,  # 上采样结果 + 边缘图
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入 x 的形状: [batch_size, in_channels, H, W]

        # 1. 提取边缘特征
        edge_map = self.edge_conv(x)  # [batch_size, 1, H, W]
        edge_map = torch.abs(edge_map)  # 取绝对值，确保边缘信息为正
        edge_map = self.relu(edge_map)

        # 2. 上采样输入图像
        upsampled = self.upconv(x)  # [batch_size, out_channels, H*upscale_factor, W*upscale_factor]
        upsampled = self.relu(upsampled)

        # 3. 上采样边缘图到相同尺寸
        edge_map_up = F.interpolate(
            edge_map,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False
        )  # [batch_size, 1, H*upscale_factor, W*upscale_factor]

        # 4. 融合边缘图和上采样结果
        combined = torch.cat([upsampled, edge_map_up], dim=1)  # [batch_size, out_channels+1, H', W']
        output = self.fusion_conv(combined)  # [batch_size, out_channels, H', W']
        output = self.relu(output)

        return output


class ShiftedChannel(nn.Module):
    def __init__(self, shift_ratio=0.25):
        super(ShiftedChannel, self).__init__()
        self.shift_ratio = shift_ratio

    def forward(self, x):
        B, C, H, W = x.shape
        shift_size = int(C * self.shift_ratio)
        x_shifted = torch.cat([
            x[:, :shift_size].roll(shifts=1, dims=2),
            x[:, shift_size:2 * shift_size].roll(shifts=-1, dims=2),
            x[:, 2 * shift_size:3 * shift_size].roll(shifts=1, dims=3),
            x[:, 3 * shift_size:],
        ], dim=1)
        return x_shifted


# 令牌化MLP块
class TokenizedMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, token_dim=64):
        super(TokenizedMLPBlock, self).__init__()
        self.shift = ShiftedChannel()
        self.to_token = nn.Conv2d(in_channels, token_dim, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, out_channels)
        )
        self.to_space = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.shift(x)
        x = self.to_token(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        x = self.mlp(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.to_space(x)
        return x


def interleave_tensors(tensor1, tensor2, dim=0):
    """
    将两个张量按照指定维度交错拼接。

    参数：
        tensor1: 第一个输入张量
        tensor2: 第二个输入张量
        dim: 指定交错的维度，默认为0

    返回：
        交错拼接后的张量

    要求：
        tensor1 和 tensor2 的形状必须相同
    """
    # 检查输入张量的形状是否相同
    if tensor1.shape != tensor2.shape:
        raise ValueError("两个张量的形状必须相同")

    # 获取张量的形状和指定维度的长度
    shape = list(tensor1.shape)
    dim_size = shape[dim]

    # 将两个张量沿着指定维度堆叠
    stacked = torch.stack([tensor1, tensor2], dim=dim + 1)  # dim+1 是为了插入一个新维度用于交错

    # 重塑张量，使得交错的元素按顺序排列
    new_shape = shape[:dim] + [shape[dim] * 2] + shape[dim + 1:]
    interleaved = stacked.reshape(new_shape)

    return interleaved


class HighFourierTransform(nn.Module):
    def __init__(self, mask_range=20):
        """
        Args:
            mask_range (int): 高通滤波器的中心掩码范围
        """
        super(HighFourierTransform, self).__init__()
        self.mask_range = mask_range

    def _create_high_pass_mask(self, rows, cols):
        """创建高通滤波掩码"""
        crow, ccol = rows // 2, cols // 2
        # 限制mask范围不超过图像尺寸
        mask_range = min(self.mask_range, min(crow, ccol))

        # 创建掩码
        mask = torch.ones((rows, cols), dtype=torch.float32)
        mask[crow - mask_range:crow + mask_range,
        ccol - mask_range:ccol + mask_range] = 0

        return mask

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, channels, height, width]

        Returns:
            torch.Tensor: 高通滤波后的张量，保持输入形状
        """
        batch_size, channels, height, width = x.shape

        # 创建高通滤波掩码并调整形状
        mask_h = self._create_high_pass_mask(height, width)
        mask_h = mask_h.view(1, 1, height, width).to(x.device)  # [1, 1, H, W]

        # 傅里叶变换
        dft = torch.fft.fft2(x)
        dft_shift = torch.fft.fftshift(dft)

        # 应用高通滤波
        fshift_h = dft_shift * mask_h
        f_ishift_h = torch.fft.ifftshift(fshift_h)

        # 逆傅里叶变换并取实部
        img_back_h = torch.abs(torch.fft.ifft2(f_ishift_h))

        return img_back_h


class Prediction_Guided_Refinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        x = x + x * torch.sigmoid(gt_pre)
        return x, torch.sigmoid(gt_pre)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class EELUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EELUnet, self).__init__()

        # 模型名称
        self.name = "eelunet"

        # 编码器部分

        self.enc1 = nn.Sequential(
            self.conv_block(in_channels, 64),
            # self.mlp_conv_block(in_channels, 64),
        )
        self.enc2 = nn.Sequential(
            self.conv_block(64, 128),
            # self.mlp_conv_block(64, 128),
        )
        self.enc3 = nn.Sequential(
            # self.conv_block(128, 256),
            self.mlp_conv_block(128, 256),
        )
        self.enc4 = nn.Sequential(
            # self.conv_block(256, 512),
            self.mlp_conv_block(256, 512),
        )

        # 瓶颈层
        self.bottleneck = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                TokenizedMLPBlock(1024, 1024),
                nn.ReLU(inplace=True),
            )
        # self.bottleneck = self.conv_block(512, 1024)

        # 解码器部分
        # self.upconv4 = self.upconv_block(1024, 512)
        self.upconv4 = self.mlp_upconv_block(1024, 512)
        # self.dec4 = Grouped_multi_axis_Hadamard_Product_Attention(512, 512)
        # self.dec4 = self.conv_block(1024, 512)
        self.dec4 = self.mlp_conv_block(1024, 512)

        # self.upconv3 = self.upconv_block(512, 256)
        self.upconv3 = self.mlp_upconv_block(512, 256)
        # self.dec3 = Grouped_multi_axis_Hadamard_Product_Attention(256, 256)
        # self.dec3 = self.conv_block(512, 256)
        self.dec3 = self.mlp_conv_block(512, 256)

        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        # self.upconv2 = self.mlp_upconv_block(256, 128)
        # self.dec2 = self.mlp_conv_block(256, 128)


        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        # self.upconv1 = self.mlp_upconv_block(128, 64)
        # self.dec1 = self.mlp_conv_block(128, 64)


        # 辅助边缘分支：利用最后一个解码器特征生成1通道边缘预测图
        self.pred5 = Prediction_Guided_Refinement(1024)
        self.pred4 = Prediction_Guided_Refinement(512)
        self.pred3 = Prediction_Guided_Refinement(256)
        self.pred2 = Prediction_Guided_Refinement(128)
        self.pred1 = Prediction_Guided_Refinement(64)

        self.edge_upconv_4 = nn.Sequential(
            # self.upconv_block(1024, 512),
            self.mlp_upconv_block(1024, 512),
            # Grouped_multi_axis_Hadamard_Product_Attention(1024, 512),
            # self.conv_block(512, 512),
            self.mlp_conv_block(512, 512)
        )
        self.edge_upconv_3 = nn.Sequential(
            # self.upconv_block(512, 256),
            self.mlp_upconv_block(512, 256),
            # Grouped_multi_axis_Hadamard_Product_Attention(512, 256)
            # self.conv_block(256, 256),
            self.mlp_conv_block(256, 256),
        )

        self.edge_upconv_2 = nn.Sequential(
            self.upconv_block(256, 128),
            # self.mlp_upconv_block(256, 128),
            HighFourierTransform(),
            self.conv_block(128, 128),
            # self.mlp_conv_block(128, 128),
        )
        self.edge_upconv_1 = nn.Sequential(
            self.upconv_block(128, 64),
            # self.mlp_upconv_block(128, 64),
            HighFourierTransform(),
            self.conv_block(64, 64),
            # self.mlp_conv_block(64, 64)
        )

        self.final = nn.Sequential(
            LayerNorm(normalized_shape=64, data_format='channels_first'),
            nn.Conv2d(64, out_channels, 1)
        )

    def conv_block(self, in_channels, out_channels):
        # 定义卷积块，包括两个卷积层、ReLU激活函数
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # Grouped_multi_axis_Hadamard_Product_Attention(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def mlp_conv_block(self, in_channels, out_channels):
        # 定义卷积块，包括两个卷积层、ReLU激活函数
        return nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            TokenizedMLPBlock(out_channels, out_channels),
            # Grouped_multi_axis_Hadamard_Product_Attention(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2)  # 添加Dropout层
        )

    def upconv_block(self, in_channels, out_channels):
        # 定义上采样块，使用反卷积层
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
        )

    def mlp_upconv_block(self, in_channels, out_channels):
        # 定义上采样块，使用反卷积层
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            TokenizedMLPBlock(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )

    def center_crop(self, layer, target_size):
        """对特征图进行中心裁剪，使其与目标大小匹配"""
        _, _, h, w = layer.size()
        target_h, target_w = target_size[2], target_size[3]
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return layer[:, :, top:top + target_h, left:left + target_w]

    def forward(self, x):
        # 主分支
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

        bottleneck, edge_5 = self.pred5(bottleneck)

        # 辅助分支
        edge_dec4 = self.edge_upconv_4(bottleneck)
        edge_dec3 = self.edge_upconv_3(edge_dec4)
        edge_dec2 = self.edge_upconv_2(edge_dec3)
        edge_dec1 = self.edge_upconv_1(edge_dec2)

        # 解码器部分
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.add(dec4, edge_dec4)
        enc4_crop = self.center_crop(enc4, dec4.size())
        # dec4 = torch.concat((dec4, enc4_crop), dim=1)
        dec4 = interleave_tensors(dec4, enc4_crop, dim=1)
        dec4 = self.dec4(dec4)

        dec3, edge_4 = self.pred4(dec4)
        dec3 = self.upconv3(dec3)
        dec3 = torch.add(dec3, edge_dec3)
        enc3_crop = self.center_crop(enc3, dec3.size())
        # dec3 = torch.concat((dec3, enc3_crop), dim=1)
        dec3 = interleave_tensors(dec3, enc3_crop, dim=1)
        dec3 = self.dec3(dec3)

        dec2, edge_3 = self.pred3(dec3)
        dec2 = self.upconv2(dec2)
        dec2 = torch.add(dec2, edge_dec2)
        enc2_crop = self.center_crop(enc2, dec2.size())
        # dec2 = torch.concat((dec2, enc2_crop), dim=1)
        dec2 = interleave_tensors(dec2, enc2_crop, dim=1)
        dec2 = self.dec2(dec2)

        dec1, edge_2 = self.pred2(dec2)
        dec1 = self.upconv1(dec1)
        dec1 = torch.add(dec1, edge_dec1)
        enc1_crop = self.center_crop(enc1, dec1.size())
        # dec1 = torch.concat((dec1, enc1_crop), dim=1)
        dec1 = interleave_tensors(dec1, enc1_crop, dim=1)
        dec1 = self.dec1(dec1)

        seg_out, edge_1 = self.pred1(dec1)

        seg_out = self.final(seg_out)

        seg_out = torch.sigmoid(seg_out)

        return seg_out, [edge_5, edge_4, edge_3, edge_2, edge_1]


# Example of how to use
# 测试模型
if __name__ == "__main__":
    model = EELUnet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    y, _ = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
