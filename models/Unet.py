import torch.nn as nn
import torch.nn.functional as F
import torch
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 模型名称
        self.name = "unet"

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

        # 最终输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        # 定义卷积块，包括两个卷积层和激活函数
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        # 定义上采样块，包括反卷积层
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

        # 通过瓶颈层
        bottleneck = self.bottleneck(bottleneck)

        # 解码器部分
        dec4 = self.upconv4(bottleneck)

        enc4 = self.center_crop(enc4, dec4.size())
        dec4 = torch.concat((dec4, enc4), dim=1)  # 跳跃连接
        dec4 = self.conv4(dec4)
        dec3 = self.upconv3(dec4)

        enc3 = self.center_crop(enc3, dec3.size())
        dec3 = torch.concat((dec3, enc3), dim=1)  # 跳跃连接
        dec3 = self.conv3(dec3)
        dec2 = self.upconv2(dec3)

        enc2 = self.center_crop(enc2, dec2.size())
        dec2 = torch.concat((dec2, enc2), dim=1)  # 跳跃连接
        dec2 = self.conv2(dec2)
        dec1 = self.upconv1(dec2)

        enc1 = self.center_crop(enc1, dec1.size())
        dev1 = torch.concat((dec1, enc1), dim=1)  # 跳跃连接
        dec1 = self.conv1(dev1)  # 跳跃连接
        # 最终输出层
        out = self.final_conv(dec1)
        return out
