import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import canny_edge_torch, visualize_images, calculate_contribution


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



class Image_Prediction_Generator(nn.Module):
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

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        # ----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # ----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # ----------dw----------#
        x4 = self.dw(x4)
        # ----------concat----------#
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # ----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x


class EdgeUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeUnet, self).__init__()

        # 模型名称
        self.name = "edgeunet"

        # 编码器部分

        self.enc1 = nn.Sequential(
            self.conv_block(in_channels, 64),
        )
        self.enc2 = nn.Sequential(
            self.conv_block(64, 128),
        )
        self.enc3 = nn.Sequential(
            self.conv_block(128, 256),
        )
        self.enc4 = nn.Sequential(
            self.conv_block(256, 512),
        )

        # 瓶颈层
        self.bottleneck = self.conv_block(512, 1024)

        # 解码器部分
        self.upconv4 = self.upconv_block(1024, 512)

        # self.conv4 = self.conv_block(1024, 512)
        self.dec4 = Grouped_multi_axis_Hadamard_Product_Attention(1024, 512)

        self.upconv3 = self.upconv_block(512, 256)

        # self.conv3 = self.conv_block(512, 256)
        self.dec3 = Grouped_multi_axis_Hadamard_Product_Attention(512, 256)

        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # 最终输出层（语义分割结果）
        self.final_conv = nn.Sequential(
            HighFourierTransform(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )


        # 辅助边缘分支：利用最后一个解码器特征生成1通道边缘预测图

        self.edge_conv_5 = nn.Conv2d(1024, 1, kernel_size=1)
        self.edge_conv_4 = nn.Conv2d(512, 1, kernel_size=1)
        self.edge_conv_3 = nn.Conv2d(256, 1, kernel_size=1)
        self.edge_conv_2 = nn.Conv2d(128, 1, kernel_size=1)
        self.edge_conv_1 = nn.Conv2d(64, 1, kernel_size=1)

        self.pred5 = Image_Prediction_Generator(1024)
        self.pred4 = Image_Prediction_Generator(512)
        self.pred3 = Image_Prediction_Generator(256)
        self.pred2 = Image_Prediction_Generator(128)
        self.pred1 = Image_Prediction_Generator(64)



        self.edge_upconv_4 = nn.Sequential(self.upconv_block(1024, 512),
                                           Grouped_multi_axis_Hadamard_Product_Attention(512, 512))
        self.edge_upconv_3 = nn.Sequential(self.upconv_block(512, 256),
                                           Grouped_multi_axis_Hadamard_Product_Attention(256, 256))

        self.edge_upconv_2 = nn.Sequential(self.upconv_block(256, 128),
                                           self.conv_block(128, 128))
        self.edge_upconv_1 = nn.Sequential(self.upconv_block(128, 64),
                                           self.conv_block(64, 64))
        # self.edge_final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.edge_final_conv = nn.Sequential(
            # HighFourierTransform(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

        # self.edge_enc1 = nn.Sequential(
        #     HighFourierTransform(),
        #     self.conv_block(in_channels, 64),
        # )
        # self.edge_enc2 = nn.Sequential(
        #     HighFourierTransform(),
        #     self.conv_block(64, 128),
        # )
        # self.edge = nn.Sequential(
        #     HighFourierTransform(),
        #     self.conv_block(128, 256),
        # )
        # self.edge_enc4 = nn.Sequential(
        #     HighFourierTransform(),
        #     self.conv_block(256, 512),
        # )
        # self.edge_bottleneck = nn.Sequential(
        #     HighFourierTransform(),
        #     self.conv_block(512, 1024)
        # )


    def conv_block(self, in_channels, out_channels):
        # 定义卷积块，包括两个卷积层、ReLU激活函数
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2)  # 添加Dropout层
        )

    def upconv_block(self, in_channels, out_channels):
        # 定义上采样块，使用反卷积层
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
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
        # 解码器部分
        dec4 = self.upconv4(bottleneck)
        enc4_crop = self.center_crop(enc4, dec4.size())
        dec4 = torch.cat((dec4, enc4_crop), dim=1)  # 跳跃连接
        dec4 = self.dec4(dec4)

        dec3, edge_4 = self.pred4(dec4)
        dec3 = self.upconv3(dec3)
        enc3_crop = self.center_crop(enc3, dec3.size())
        dec3 = torch.cat((dec3, enc3_crop), dim=1)  # 跳跃连接
        dec3 = self.dec3(dec3)

        dec2, edge_3 = self.pred3(dec3)
        dec2 = self.upconv2(dec2)
        enc2_crop = self.center_crop(enc2, dec2.size())
        dec2 = torch.cat((dec2, enc2_crop), dim=1)  # 跳跃连接
        dec2 = self.dec2(dec2)

        dec1, edge_2 = self.pred2(dec2)
        dec1 = self.upconv1(dec1)
        enc1_crop = self.center_crop(enc1, dec1.size())
        dec1 = torch.cat((dec1, enc1_crop), dim=1)  # 跳跃连接
        dec1 = self.dec1(dec1)

        seg_out, edge_1 = self.pred1(dec1)
        seg_out = self.final_conv(dec1)
        # 主分支输出：语义分割结果
        seg_out = torch.sigmoid(seg_out)


        # 辅助分支
        # edge_enc1 = self.edge_enc1(x)
        # edge_enc2 = nn.MaxPool2d(kernel_size=2)(edge_enc1)
        # edge_enc2 = self.edge_enc2(edge_enc2)
        # edge_enc3 = nn.MaxPool2d(kernel_size=2)(edge_enc2)
        # edge_enc3 = self.edge(edge_enc3)
        # edge_enc4 = nn.MaxPool2d(kernel_size=2)(edge_enc3)
        # edge_enc4 = self.edge_enc4(edge_enc4)
        # edge_bottleneck = nn.MaxPool2d(kernel_size=2)(edge_enc4)
        # edge_bottleneck = self.edge_bottleneck(bottleneck)
        edge_dec5 = self.edge_upconv_4(bottleneck)
        edge_dec4 = self.edge_upconv_3(edge_dec5)
        edge_dec3 = self.edge_upconv_2(edge_dec4)
        edge_dec2 = self.edge_upconv_1(edge_dec3)

        edge_out = self.edge_final_conv(edge_dec2)
        # visualize_images(edge_out.cpu().detach().numpy(), "first")
        # edge_out = HighFourierTransform()(edge_out)
        # visualize_images(edge_out.cpu().detach().numpy(), "second")
        edge_out = torch.sigmoid(edge_out)
        # visualize_images(edge_out.cpu().detach().numpy(), "third")

        # visualize_images(edge_out.cpu().detach().numpy(), "edge_out")
        # visualize_images(seg_out.cpu().detach().numpy(), "seg_out")
        # seg_out = torch.min(seg_out + edge_out, torch.ones_like(seg_out))
        # visualize_images(seg_out.cpu().detach().numpy(), "seg_out")
        calculate_contribution(seg_out, edge_out)
        seg_out = torch.max(seg_out, edge_out)

        return seg_out, [edge_5, edge_4, edge_3, edge_2, edge_1]
    # def forward(self, x):
    #     # 编码器部分
    #     enc1 = self.enc1(x)
    #     enc2 = nn.MaxPool2d(kernel_size=2)(enc1)
    #     enc2 = self.enc2(enc2)
    #     enc3 = nn.MaxPool2d(kernel_size=2)(enc2)
    #     enc3 = self.enc3(enc3)
    #     enc4 = nn.MaxPool2d(kernel_size=2)(enc3)
    #     enc4 = self.enc4(enc4)
    #     bottleneck = nn.MaxPool2d(kernel_size=2)(enc4)
    #     bottleneck = self.bottleneck(bottleneck)
    #
    #     # 解码器部分
    #     dec4 = self.upconv4(bottleneck)
    #     enc4_crop = self.center_crop(enc4, dec4.size())
    #     dec4 = torch.cat((dec4, enc4_crop), dim=1)  # 跳跃连接
    #     dec4 = self.conv4(dec4)
    #
    #     dec3 = self.upconv3(dec4)
    #     enc3_crop = self.center_crop(enc3, dec3.size())
    #     dec3 = torch.cat((dec3, enc3_crop), dim=1)  # 跳跃连接
    #     dec3 = self.conv3(dec3)
    #
    #     dec2 = self.upconv2(dec3)
    #     enc2_crop = self.center_crop(enc2, dec2.size())
    #     dec2 = torch.cat((dec2, enc2_crop), dim=1)  # 跳跃连接
    #     dec2 = self.conv2(dec2)
    #
    #     dec1 = self.upconv1(dec2)
    #     enc1_crop = self.center_crop(enc1, dec1.size())
    #     dec1 = torch.cat((dec1, enc1_crop), dim=1)  # 跳跃连接
    #     dec1 = self.conv1(dec1)
    #
    #     # 主分支输出：语义分割结果
    #     seg_out = self.final_conv(dec1)
    #     seg_out = torch.sigmoid(seg_out)
    #
    #     edge_5 = self.edge_conv_5(bottleneck)
    #     edge_5 = torch.sigmoid(edge_5)
    #
    #     edge_4 = self.edge_conv_4(dec4)
    #     edge_4 = torch.sigmoid(edge_4)
    #
    #     edge_3 = self.edge_conv_3(dec3)
    #     edge_3 = torch.sigmoid(edge_3)
    #
    #     edge_2 = self.edge_conv_2(dec2)
    #     edge_2 = torch.sigmoid(edge_2)
    #
    #     edge_1 = self.edge_conv_1(dec1)
    #     edge_1 = torch.sigmoid(edge_1)
    #
    #     edge_dec5 = self.edge_upconv_4(bottleneck)
    #     edge_dec4 = self.edge_upconv_3(edge_dec5)
    #     edge_dec3 = self.edge_upconv_2(edge_dec4)
    #     edge_dec2 = self.edge_upconv_1(edge_dec3)
    #
    #     edge_out = self.edge_final_conv(edge_dec2)
    #     edge_out = torch.sigmoid(edge_out)
    #
    #     seg_out = torch.max(seg_out, edge_out)
    #
    #     return seg_out, [edge_5, edge_4, edge_3, edge_2, edge_1]
    # def forward(self, x):
    #     # 编码器部分
    #     enc1 = self.enc1(x)
    #     enc2 = nn.MaxPool2d(kernel_size=2)(enc1)
    #     enc2 = self.enc2(enc2)
    #     enc3 = nn.MaxPool2d(kernel_size=2)(enc2)
    #     enc3 = self.enc3(enc3)
    #     enc4 = nn.MaxPool2d(kernel_size=2)(enc3)
    #     enc4 = self.enc4(enc4)
    #     bottleneck = nn.MaxPool2d(kernel_size=2)(enc4)
    #     bottleneck = self.bottleneck(bottleneck)
    #
    #     # 解码器部分
    #     dec4 = self.upconv4(bottleneck)
    #     enc4_crop = self.center_crop(enc4, dec4.size())
    #     dec4 = torch.cat((dec4, enc4_crop), dim=1)  # 跳跃连接
    #     dec4 = self.conv4(dec4)
    #
    #     dec3 = self.upconv3(dec4)
    #     enc3_crop = self.center_crop(enc3, dec3.size())
    #     dec3 = torch.cat((dec3, enc3_crop), dim=1)  # 跳跃连接
    #     dec3 = self.conv3(dec3)
    #
    #     dec2 = self.upconv2(dec3)
    #     enc2_crop = self.center_crop(enc2, dec2.size())
    #     dec2 = torch.cat((dec2, enc2_crop), dim=1)  # 跳跃连接
    #     dec2 = self.conv2(dec2)
    #
    #     dec1 = self.upconv1(dec2)
    #     enc1_crop = self.center_crop(enc1, dec1.size())
    #     dec1 = torch.cat((dec1, enc1_crop), dim=1)  # 跳跃连接
    #     dec1 = self.conv1(dec1)
    #
    #     # 主分支输出：语义分割结果
    #     seg_out = self.final_conv(dec1)
    #     seg_out = torch.sigmoid(seg_out)
    #
    #     # 辅助分支输出：边缘预测
    #     edge_5 = self.edge_conv_5(bottleneck)
    #     edge_5 = torch.sigmoid(edge_5)
    #
    #     edge_4 = self.edge_conv_4(dec4)
    #     edge_4 = torch.sigmoid(edge_4)
    #
    #     edge_3 = self.edge_conv_3(dec3)
    #     edge_3 = torch.sigmoid(edge_3)
    #
    #     edge_2 = self.edge_conv_2(dec2)
    #     edge_2 = torch.sigmoid(edge_2)
    #
    #     edge_1 = self.edge_conv_1(dec1)
    #     edge_1 = torch.sigmoid(edge_1)
    #
    #     edge_out = self.edge_final_conv(dec1)
    #     edge_out = torch.sigmoid(edge_out)
    #
    #     # seg_out与edge_out取值范围均为 [0,1]，将edge_out中比seg_out更大的值赋值给seg_out
    #     seg_out = torch.max(seg_out, edge_out)
    #
    #
    #
    #     # binary_edge_out = canny_edge_torch_improve(binary_edge_out)
    #     # binary_edge_out = generate_edge_label(binary_edge_out.cpu().numpy())
    #
    #
    #
    #
    #     # return seg_out, binary_edge_out
    #     return seg_out, [edge_5, edge_4, edge_3, edge_2, edge_1]


# Example of how to use
if __name__ == "__main__":
    model = EdgeUnet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)  # Example input tensor
    seg_out, edge_outs = model(x)
    # print(seg_out.shape) # Expected to be [1, 1, 256, 256]
    # print(edge_outs[0].shape) # Expected to be [1, 1, 16, 16]
