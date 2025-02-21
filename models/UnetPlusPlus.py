import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetPlusPlus, self).__init__()

        # 编码器部分 (Encoder)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 解码器部分 (Decoder)
        self.upconv4 = self.upconv_block(512, 256)
        self.upconv3 = self.upconv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)
        self.upconv1 = self.upconv_block(64, out_channels)

        # 中间层连接部分 (Bottleneck)
        self.bottleneck = self.conv_block(512, 512)

        # 用来调整通道数的卷积层
        self.conv_up3 = self.conv_block(512, 256)
        self.conv_up2 = self.conv_block(256, 128)
        self.conv_up1 = self.conv_block(128, 64)

        # 密集跳跃连接部分 (Dense Skip Connections)
        self.enc2_dec4 = self.conv_block(128 + 256, 256)  # 连接后的通道数修正
        self.enc3_dec4 = self.conv_block(256 + 256, 512)  # 连接后的通道数修正
        self.enc3_dec3 = self.conv_block(256 + 128, 256)  # 连接后的通道数修正
        self.enc2_dec3 = self.conv_block(128 + 64, 128)  # 连接后的通道数修正

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码部分
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # 中间瓶颈部分
        bottleneck = self.bottleneck(enc4)

        # 解码部分
        dec4 = self.upconv4(bottleneck)
        dec4 = self.crop_and_concat(dec4, enc3)  # U-Net++跳跃连接
        dec4 = self.enc2_dec4(dec4)  # 额外跳跃连接

        dec3 = self.upconv3(dec4)
        dec3 = self.crop_and_concat(dec3, enc2)  # U-Net++跳跃连接
        dec3 = self.enc3_dec4(dec3)  # 额外跳跃连接
        dec3 = self.enc3_dec3(dec3)  # 额外跳跃连接

        dec2 = self.upconv2(dec3)
        dec2 = self.crop_and_concat(dec2, enc1)  # U-Net++跳跃连接
        dec2 = self.enc2_dec3(dec2)  # 额外跳跃连接

        dec1 = self.upconv1(dec2)

        return dec1

    def crop_and_concat(self, dec, enc):
        """裁剪并拼接解码器和编码器的输出，使其尺寸匹配"""
        _, _, h, w = enc.size()
        dec = dec[:, :, :h, :w]  # 裁剪解码器的输出
        return torch.cat([dec, enc], dim=1)


# Example of how to use
if __name__ == "__main__":
    model = UNetPlusPlus(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 256, 256)  # Example input tensor
    output = model(x)
    print(output.shape)  # Expected to be [1, 1, 256, 256]
