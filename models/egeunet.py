import torch
from torch import nn
import torch.nn.functional as F
# from einops import rearrange

from timm.models.layers import trunc_normal_
import math



class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

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

# 
# class group_aggregation_bridge(nn.Module):
#     def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
#         super().__init__()
#         self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
#         group_size = dim_xl // 2
#         self.g0 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
#             nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
#                       padding=(k_size+(k_size-1)*(d_list[0]-1))//2,
#                       dilation=d_list[0], groups=group_size + 1)
#         )
#         self.g1 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
#             nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
#                       padding=(k_size+(k_size-1)*(d_list[1]-1))//2,
#                       dilation=d_list[1], groups=group_size + 1)
#         )
#         self.g2 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
#             nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
#                       padding=(k_size+(k_size-1)*(d_list[2]-1))//2,
#                       dilation=d_list[2], groups=group_size + 1)
#         )
#         self.g3 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
#             nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
#                       padding=(k_size+(k_size-1)*(d_list[3]-1))//2,
#                       dilation=d_list[3], groups=group_size + 1)
#         )
#         self.tail_conv = nn.Sequential(
#             LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
#             nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
#         )
#     def forward(self, xh, xl, mask):
#         xh = self.pre_project(xh)
#         xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
#         xh = torch.chunk(xh, 4, dim=1)
#         xl = torch.chunk(xl, 4, dim=1)
#         x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
#         x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
#         x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
#         x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
#         x = torch.cat((x0,x1,x2,x3), dim=1)
#         x = self.tail_conv(x)
#         return x


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


class ConvLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim,
                               padding_mode='reflect')  # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0, stride=1)
        self.act1 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0, stride=1)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.bn(x))


class Image_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        x = x + x * torch.sigmoid(gt_pre)
        return x, gt_pre


class Merge(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

    def forward(self, x1, x2, gt_pre, w):
        x = x1 + x2 + torch.sigmoid(gt_pre) * x2 * w
        return x


class EGEUNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True):
        super().__init__()

        self.name = "egeunet"

        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=1, padding=1),
            ConvLayer(24),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(24, 32),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(32, 48),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(48, 64),
        )

        self.Down1 = Down(8)
        self.Down2 = Down(16)
        self.Down3 = Down(24)

        self.merge1 = Merge(8)
        self.merge2 = Merge(16)
        self.merge3 = Merge(24)
        self.merge4 = Merge(32)
        self.merge5 = Merge(48)

        self.pred1 = Image_Prediction_Generator(48)
        self.pred2 = Image_Prediction_Generator(32)
        self.pred3 = Image_Prediction_Generator(24)
        self.pred4 = Image_Prediction_Generator(16)
        self.pred5 = Image_Prediction_Generator(8)

        # if bridge:
        #     self.GAB1 = group_aggregation_bridge(16, 8)
        #     self.GAB2 = group_aggregation_bridge(24, 16)
        #     self.GAB3 = group_aggregation_bridge(32, 24)
        #     self.GAB4 = group_aggregation_bridge(48, 32)
        #     self.GAB5 = group_aggregation_bridge(64, 48)
        #     print('group_aggregation_bridge was used')
        # if gt_ds:
        #     self.gt_conv1 = nn.Sequential(nn.Conv2d(48, 1, 1))
        #     self.gt_conv2 = nn.Sequential(nn.Conv2d(32, 1, 1))
        #     self.gt_conv3 = nn.Sequential(nn.Conv2d(24, 1, 1))
        #     self.gt_conv4 = nn.Sequential(nn.Conv2d(16, 1, 1))
        #     self.gt_conv5 = nn.Sequential(nn.Conv2d(8, 1, 1))
        #     print('gt deep supervision was used')

        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(64, 48),
        )
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(48, 32),
        )
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(32, 24),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(24, 16, 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, 8)
        self.ebn2 = nn.GroupNorm(4, 16)
        self.ebn3 = nn.GroupNorm(4, 24)
        self.ebn4 = nn.GroupNorm(4, 32)
        self.ebn5 = nn.GroupNorm(4, 48)
        self.dbn1 = nn.GroupNorm(4, 48)
        self.dbn2 = nn.GroupNorm(4, 32)
        self.dbn3 = nn.GroupNorm(4, 24)
        self.dbn4 = nn.GroupNorm(4, 16)
        self.dbn5 = nn.GroupNorm(4, 8)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = self.encoder1(x)
        out = F.gelu(self.Down1(self.ebn1(out)))
        t1 = out  # b, 8, 128, 128

        out = self.encoder2(out)
        out = F.gelu(self.Down2(self.ebn2(out)))
        t2 = out  # b, 16, 64, 64

        out = self.encoder3(out)
        out = F.gelu(self.Down3(self.ebn3(out)))
        t3 = out  # b, 24, 32, 32

        out = self.encoder4(out)
        out = F.gelu(F.max_pool2d(self.ebn4(out), 2))
        t4 = out  # b, 32, 16, 16

        out = self.encoder5(out)
        out = F.gelu(F.max_pool2d(self.ebn5(out), 2))
        t5 = out  # b, 48, 8, 8

        out = self.encoder6(out)
        out = F.gelu(out)  # b, 64, 8, 8

        out = self.decoder1(out)
        out = F.gelu(self.dbn1(out))  # b, 48, 8, 8

        out, gt_pre5 = self.pred1(out)
        out = self.merge5(out, t5, gt_pre5, 0.1)  # b, 48, 8, 8
        gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)

        out = self.decoder2(out)
        out = F.gelu(
            F.interpolate(self.dbn2(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, 32, 16, 16
        out, gt_pre4 = self.pred2(out)
        out = self.merge4(out, t4, gt_pre4, 0.2)  # b, 32, 16, 16
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)

        out = self.decoder3(out)

        out = F.gelu(
            F.interpolate(self.dbn3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, 24, 32, 32
        out, gt_pre3 = self.pred3(out)
        out = self.merge3(out, t3, gt_pre3, 0.3)  # b, 24, 32, 32
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)

        out = self.decoder4(out)
        out = F.gelu(
            F.interpolate(self.dbn4(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, 16, 64, 64
        out, gt_pre2 = self.pred4(out)
        out = self.merge2(out, t2, gt_pre2, 0.4)  # b, 16, 64, 64
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)

        out = self.decoder5(out)
        out = F.gelu(
            F.interpolate(self.dbn5(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, 8, 128, 128
        out, gt_pre1 = self.pred5(out)
        out = self.merge1(out, t1, gt_pre1, 0.5)  # b, 3, 128, 128
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.final(out)
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, num_class, H, W

        if self.gt_ds:
            return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2),
                    torch.sigmoid(gt_pre1)), torch.sigmoid(out)
        else:
            return torch.sigmoid(out)

class EGEUNet_Large(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True):
        super().__init__()

        self.name = "egeunet"

        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=1, padding=1),
            ConvLayer(24),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(24, 32),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(32, 48),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(48, 64),
        )

        self.Down1 = Down(8)
        self.Down2 = Down(16)
        self.Down3 = Down(24)

        self.merge1 = Merge(8)
        self.merge2 = Merge(16)
        self.merge3 = Merge(24)
        self.merge4 = Merge(32)
        self.merge5 = Merge(48)

        self.pred1 = Image_Prediction_Generator(48)
        self.pred2 = Image_Prediction_Generator(32)
        self.pred3 = Image_Prediction_Generator(24)
        self.pred4 = Image_Prediction_Generator(16)
        self.pred5 = Image_Prediction_Generator(8)

        # if bridge:
        #     self.GAB1 = group_aggregation_bridge(16, 8)
        #     self.GAB2 = group_aggregation_bridge(24, 16)
        #     self.GAB3 = group_aggregation_bridge(32, 24)
        #     self.GAB4 = group_aggregation_bridge(48, 32)
        #     self.GAB5 = group_aggregation_bridge(64, 48)
        #     print('group_aggregation_bridge was used')
        # if gt_ds:
        #     self.gt_conv1 = nn.Sequential(nn.Conv2d(48, 1, 1))
        #     self.gt_conv2 = nn.Sequential(nn.Conv2d(32, 1, 1))
        #     self.gt_conv3 = nn.Sequential(nn.Conv2d(24, 1, 1))
        #     self.gt_conv4 = nn.Sequential(nn.Conv2d(16, 1, 1))
        #     self.gt_conv5 = nn.Sequential(nn.Conv2d(8, 1, 1))
        #     print('gt deep supervision was used')

        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(64, 48),
        )
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(48, 32),
        )
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(32, 24),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(24, 16, 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, 8)
        self.ebn2 = nn.GroupNorm(4, 16)
        self.ebn3 = nn.GroupNorm(4, 24)
        self.ebn4 = nn.GroupNorm(4, 32)
        self.ebn5 = nn.GroupNorm(4, 48)
        self.dbn1 = nn.GroupNorm(4, 48)
        self.dbn2 = nn.GroupNorm(4, 32)
        self.dbn3 = nn.GroupNorm(4, 24)
        self.dbn4 = nn.GroupNorm(4, 16)
        self.dbn5 = nn.GroupNorm(4, 8)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = self.encoder1(x)
        out = F.gelu(self.Down1(self.ebn1(out)))
        t1 = out  # b, 8, 128, 128

        out = self.encoder2(out)
        out = F.gelu(self.Down2(self.ebn2(out)))
        t2 = out  # b, 16, 64, 64

        out = self.encoder3(out)
        out = F.gelu(self.Down3(self.ebn3(out)))
        t3 = out  # b, 24, 32, 32

        out = self.encoder4(out)
        out = F.gelu(F.max_pool2d(self.ebn4(out), 2))
        t4 = out  # b, 32, 16, 16

        out = self.encoder5(out)
        out = F.gelu(F.max_pool2d(self.ebn5(out), 2))
        t5 = out  # b, 48, 8, 8

        out = self.encoder6(out)
        out = F.gelu(out)  # b, 64, 8, 8

        out = self.decoder1(out)
        out = F.gelu(self.dbn1(out))  # b, 48, 8, 8

        out, gt_pre5 = self.pred1(out)
        out = self.merge5(out, t5, gt_pre5, 0.1)  # b, 48, 8, 8
        gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)

        out = self.decoder2(out)
        out = F.gelu(
            F.interpolate(self.dbn2(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, 32, 16, 16
        out, gt_pre4 = self.pred2(out)
        out = self.merge4(out, t4, gt_pre4, 0.2)  # b, 32, 16, 16
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)

        out = self.decoder3(out)

        out = F.gelu(
            F.interpolate(self.dbn3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, 24, 32, 32
        out, gt_pre3 = self.pred3(out)
        out = self.merge3(out, t3, gt_pre3, 0.3)  # b, 24, 32, 32
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)

        out = self.decoder4(out)
        out = F.gelu(
            F.interpolate(self.dbn4(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, 16, 64, 64
        out, gt_pre2 = self.pred4(out)
        out = self.merge2(out, t2, gt_pre2, 0.4)  # b, 16, 64, 64
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)

        out = self.decoder5(out)
        out = F.gelu(
            F.interpolate(self.dbn5(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, 8, 128, 128
        out, gt_pre1 = self.pred5(out)
        out = self.merge1(out, t1, gt_pre1, 0.5)  # b, 3, 128, 128
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.final(out)
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, num_class, H, W

        if self.gt_ds:
            return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2),
                    torch.sigmoid(gt_pre1)), torch.sigmoid(out)
        else:
            return torch.sigmoid(out)