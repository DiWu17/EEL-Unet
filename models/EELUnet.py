import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils.tools import canny_edge_torch, visualize_images, calculate_contribution, visualize_feature_maps


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    通过全局池化压缩空间信息，然后用全连接层学习通道间的关系
    """

    def __init__(self, in_channels, reduction=16):
        """
        参数：
            in_channels (int): 输入特征的通道数
            reduction (int): 降维比例，用于减少参数量，默认16
        """
        super(ChannelAttention, self).__init__()

        self.in_channels = in_channels
        self.reduction = reduction

        # 全局平均池化，将空间维度压缩为1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 降维全连接层
        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            bias=True
        )

        # 升维全连接层，恢复到原始通道数
        self.fc2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            bias=True
        )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        参数：
            x (torch.Tensor): 输入张量，形状为 [batch_size, in_channels, height, width]
        返回：
            torch.Tensor: 加权后的特征张量，形状与输入相同
        """
        # 输入形状：[batch_size, in_channels, height, width]
        batch_size, channels, _, _ = x.size()

        # 1. 全局平均池化，压缩空间维度
        # 输出形状：[batch_size, in_channels, 1, 1]
        avg_out = self.global_avg_pool(x)

        # 2. 第一个全连接层降维
        # 输出形状：[batch_size, in_channels//reduction, 1, 1]
        fc1_out = self.fc1(avg_out)
        fc1_out = self.relu(fc1_out)

        # 3. 第二个全连接层升维
        # 输出形状：[batch_size, in_channels, 1, 1]
        fc2_out = self.fc2(fc1_out)

        # 4. Sigmoid激活，生成通道注意力权重
        # 输出形状：[batch_size, in_channels, 1, 1]
        attention_weights = self.sigmoid(fc2_out)

        # 5. 将注意力权重应用到原始输入上
        # 通过广播机制，权重会自动扩展到与输入相同的空间维度
        out = x * attention_weights

        return out


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
class ChannelAwarePatchedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, token_dim=64):
        super(ChannelAwarePatchedMLP, self).__init__()
        self.shift = ShiftedChannel()
        self.to_patch = nn.Conv2d(in_channels, token_dim, kernel_size=1)
        self.channel_attention = ChannelAttention(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, out_channels)
        )
        self.to_space = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.shift(x)
        x = self.to_patch(x)
        x = self.channel_attention(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        x = self.mlp(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.to_space(x)
        return x



class FeatureInterleaveBridge(nn.Module):
    def __init__(self, channels):
        super(FeatureInterleaveBridge, self).__init__()
        self.channels = channels
    #
    def forward(self, x1, x2, dim=1):
        # 获取张量的形状和指定维度的长度
        shape = list(x1.shape)
        # 将两个张量沿着指定维度堆叠
        stacked = torch.stack([x1, x2], dim=dim + 1)  # dim+1 是为了插入一个新维度用于交错

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


class PredictionGuidedRefinement(nn.Module):
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
            ChannelAwarePatchedMLP(1024, 1024),
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
        self.pred5 = PredictionGuidedRefinement(1024)
        self.pred4 = PredictionGuidedRefinement(512)
        self.pred3 = PredictionGuidedRefinement(256)
        self.pred2 = PredictionGuidedRefinement(128)
        self.pred1 = PredictionGuidedRefinement(64)

        self.channel_interleave_bridge4 = FeatureInterleaveBridge(1024)
        self.channel_interleave_bridge3 = FeatureInterleaveBridge(512)
        self.channel_interleave_bridge2 = FeatureInterleaveBridge(256)
        self.channel_interleave_bridge1 = FeatureInterleaveBridge(128)


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
            ChannelAwarePatchedMLP(out_channels, out_channels),
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
            ChannelAwarePatchedMLP(out_channels, out_channels),
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

        visualize_feature_maps(enc1, title="Encoder 1 Feature Maps", num_cols=8, save_path="encoder_1_feature_maps.png")

        enc2 = nn.MaxPool2d(kernel_size=2)(enc1)
        enc2 = self.enc2(enc2)

        visualize_feature_maps(enc2, title="Encoder 2 Feature Maps", num_cols=8, save_path="encoder_2_feature_maps.png")

        enc3 = nn.MaxPool2d(kernel_size=2)(enc2)
        enc3 = self.enc3(enc3)

        visualize_feature_maps(enc3, title="Encoder 3 Feature Maps", num_cols=8, save_path="encoder_3_feature_maps.png")

        enc4 = nn.MaxPool2d(kernel_size=2)(enc3)
        enc4 = self.enc4(enc4)

        visualize_feature_maps(enc4, title="Encoder 4 Feature Maps", num_cols=8, save_path="encoder_4_feature_maps.png")

        bottleneck = nn.MaxPool2d(kernel_size=2)(enc4)
        bottleneck = self.bottleneck(bottleneck)
        # print(bottleneck.shape)

        bottleneck, edge_5 = self.pred5(bottleneck)

        visualize_feature_maps(bottleneck, title="Bottleneck Feature Maps", num_cols=8, save_path="bottleneck_feature_maps.png")

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
        # dec4 = interleave_tensors(dec4, enc4_crop, dim=1)
        dec4 = self.channel_interleave_bridge4(dec4, enc4_crop, dim=1)
        dec4 = self.dec4(dec4)

        visualize_feature_maps(dec4, title="Decoder 4 Feature Maps", num_cols=8, save_path="decoder_4_feature_maps.png")

        dec3, edge_4 = self.pred4(dec4)
        dec3 = self.upconv3(dec3)
        dec3 = torch.add(dec3, edge_dec3)
        enc3_crop = self.center_crop(enc3, dec3.size())
        # dec3 = torch.concat((dec3, enc3_crop), dim=1)
        # dec3 = interleave_tensors(dec3, enc3_crop, dim=1)
        dec3 = self.channel_interleave_bridge3(dec3, enc3_crop)
        dec3 = self.dec3(dec3)

        visualize_feature_maps(dec3, title="Decoder 3 Feature Maps", num_cols=8, save_path="decoder_3_feature_maps.png")

        dec2, edge_3 = self.pred3(dec3)
        dec2 = self.upconv2(dec2)
        dec2 = torch.add(dec2, edge_dec2)
        enc2_crop = self.center_crop(enc2, dec2.size())
        # dec2 = torch.concat((dec2, enc2_crop), dim=1)
        # dec2 = interleave_tensors(dec2, enc2_crop, dim=1)
        dec2 = self.channel_interleave_bridge2(dec2, enc2_crop, dim=1)
        dec2 = self.dec2(dec2)

        visualize_feature_maps(dec2, title="Decoder 2 Feature Maps", num_cols=8, save_path="decoder_2_feature_maps.png")

        dec1, edge_2 = self.pred2(dec2)
        dec1 = self.upconv1(dec1)
        dec1 = torch.add(dec1, edge_dec1)
        enc1_crop = self.center_crop(enc1, dec1.size())
        # dec1 = torch.concat((dec1, enc1_crop), dim=1)
        # dec1 = interleave_tensors(dec1, enc1_crop, dim=1)
        dec1 = self.channel_interleave_bridge1(dec1, enc1_crop, dim=1)
        dec1 = self.dec1(dec1)

        visualize_feature_maps(dec1, title="Decoder 1 Feature Maps", num_cols=8, save_path="decoder_1_feature_maps.png")


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
