import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import canny_edge_torch, visualize_images

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
        # self.enc3 = self.conv_block(128, 256)
        # self.enc4 = self.conv_block(256, 512)
        self.enc3 = Grouped_multi_axis_Hadamard_Product_Attention(128, 256)
        self.enc4 = Grouped_multi_axis_Hadamard_Product_Attention(256, 512)

        # 瓶颈层
        self.bottleneck = self.conv_block(512, 1024)

        # 解码器部分
        self.upconv4 = self.upconv_block(1024, 512)

        # self.conv4 = self.conv_block(1024, 512)
        self.conv4 = Grouped_multi_axis_Hadamard_Product_Attention(1024, 512)

        self.upconv3 = self.upconv_block(512, 256)

        # self.conv3 = self.conv_block(512, 256)
        self.conv3 = Grouped_multi_axis_Hadamard_Product_Attention(512, 256)

        self.upconv2 = self.upconv_block(256, 128)
        self.conv2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.conv1 = self.conv_block(128, 64)

        # 最终输出层（语义分割结果）
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.edge_final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # 辅助边缘分支：利用最后一个解码器特征生成1通道边缘预测图

        self.edge_conv_5 = nn.Conv2d(1024, 1, kernel_size=1)
        self.edge_conv_4 = nn.Conv2d(512, 1, kernel_size=1)
        self.edge_conv_3 = nn.Conv2d(256, 1, kernel_size=1)
        self.edge_conv_2 = nn.Conv2d(128, 1, kernel_size=1)
        self.edge_conv_1 = nn.Conv2d(64, 1, kernel_size=1)

        # self.edge_upconv_4 = nn.Sequential(self.upconv_block(1024, 512),
        #                                    self.conv_block(512, 512))
        # self.edge_upconv_3 = nn.Sequential(self.upconv_block(512, 256),
        #                                    self.conv_block(256, 256))
        self.edge_upconv_4 = nn.Sequential(self.upconv_block(1024, 512),
                                           Grouped_multi_axis_Hadamard_Product_Attention(512, 512))
        self.edge_upconv_3 = nn.Sequential(self.upconv_block(512, 256),
                                           Grouped_multi_axis_Hadamard_Product_Attention(256, 256))

        self.edge_upconv_2 = nn.Sequential(self.upconv_block(256, 128),
                                           self.conv_block(128, 128))
        self.edge_upconv_1 = nn.Sequential(self.upconv_block(128, 64),
                                           self.conv_block(64, 64))

        self.pred5 = Image_Prediction_Generator(1024)
        self.pred4 = Image_Prediction_Generator(512)
        self.pred3 = Image_Prediction_Generator(256)
        self.pred2 = Image_Prediction_Generator(128)
        self.pred1 = Image_Prediction_Generator(64)



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
        dec4 = self.conv4(dec4)

        dec3, edge_4 = self.pred4(dec4)
        dec3 = self.upconv3(dec3)
        enc3_crop = self.center_crop(enc3, dec3.size())
        dec3 = torch.cat((dec3, enc3_crop), dim=1)  # 跳跃连接
        dec3 = self.conv3(dec3)

        dec2, edge_3 = self.pred3(dec3)
        dec2 = self.upconv2(dec2)
        enc2_crop = self.center_crop(enc2, dec2.size())
        dec2 = torch.cat((dec2, enc2_crop), dim=1)  # 跳跃连接
        dec2 = self.conv2(dec2)

        dec1, edge_2 = self.pred2(dec2)
        dec1 = self.upconv1(dec1)
        enc1_crop = self.center_crop(enc1, dec1.size())
        dec1 = torch.cat((dec1, enc1_crop), dim=1)  # 跳跃连接
        dec1 = self.conv1(dec1)

        seg_out, edge_1 = self.pred1(dec1)
        seg_out = self.final_conv(dec1)
        # 主分支输出：语义分割结果
        seg_out = torch.sigmoid(seg_out)

        edge_dec5 = self.edge_upconv_4(bottleneck)
        edge_dec4 = self.edge_upconv_3(edge_dec5)
        edge_dec3 = self.edge_upconv_2(edge_dec4)
        edge_dec2 = self.edge_upconv_1(edge_dec3)

        edge_out = self.edge_final_conv(edge_dec2)
        edge_out = torch.sigmoid(edge_out)

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
