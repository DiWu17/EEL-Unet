import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_distance_map(target_np):
    """
    根据真实二值 mask 计算距离地图。
    先利用二值腐蚀计算出 mask 的边界，再对边界的反集进行距离变换。

    :param target_np: numpy 数组，尺寸为 (H, W)，二值图（前景为1，背景为0）
    :return: 距离地图，尺寸为 (H, W)
    """
    # 将目标转换为布尔型
    target_bool = target_np.astype(np.bool_)
    # 计算腐蚀（使用 3x3 结构元素）
    eroded = binary_erosion(target_bool, structure=np.ones((3, 3)))
    # 边界为目标与腐蚀后结果的异或
    boundary = target_bool ^ eroded
    # 对边界的反集（即非边界区域）计算距离变换
    d_map = distance_transform_edt(~boundary)
    return d_map


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)



class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        # print(pred.size(), target.size())
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + \
                  self.bcedice(gt_pre4, target) * 0.2 + \
                  self.bcedice(gt_pre3, target) * 0.3 + \
                  self.bcedice(gt_pre2, target) * 0.4 + \
                  self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss

class edge_bacediceloss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre

        target_5 = F.max_pool2d(target, kernel_size=16, stride=16)
        target_4 = F.max_pool2d(target, kernel_size=8, stride=8)
        target_3 = F.max_pool2d(target, kernel_size=4, stride=4)
        target_2 = F.max_pool2d(target, kernel_size=2, stride=2)
        target_1 = target

        gt_loss = self.bcedice(gt_pre5, target_5) * 0.1 + \
                  self.bcedice(gt_pre4, target_4) * 0.2 + \
                  self.bcedice(gt_pre3, target_3) * 0.3 + \
                  self.bcedice(gt_pre2, target_2) * 0.4 + \
                  self.bcedice(gt_pre1, target_1) * 0.5
        return bcediceloss + gt_loss



class BoundaryLoss(nn.Module):
    def __init__(self):
        """
        Boundary Loss：基于真实边界距离地图对预测概率进行加权，
        越远离真实边界的像素如果被预测为正（前景）会产生更高的损失，从而鼓励
        模型在边界附近的预测更精确。
        """
        super(BoundaryLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits, target):
        """
        :param logits: 模型输出 logits，尺寸为 [B, 1, H, W]
        :param target: 真实二值 mask，尺寸为 [B, 1, H, W]，值为0或1
        :return: Boundary Loss（标量）
        """
        # 首先对 logits 进行 Sigmoid 激活得到预测概率
        pred = self.sigmoid(logits)  # 结果范围 [0, 1]
        loss = 0.0
        batch_size = pred.size(0)
        # 遍历每个样本
        for i in range(batch_size):
            # 将第 i 个样本的真实 mask 转换为 numpy 数组，形状 (H, W)
            target_np = target[i, 0].cpu().numpy().astype(np.uint8)
            # 计算距离地图，尺寸 (H, W)
            d_map = compute_distance_map(target_np)
            # 转换为 torch.Tensor，并移到与预测相同的设备
            d_map = torch.from_numpy(d_map).float().to(pred.device)
            # 归一化距离地图，防止数值过大
            d_map = d_map / (d_map.max() + 1e-7)
            # 计算当前样本的损失：
            # 将预测概率与距离地图点乘，然后除以距离地图的和，避免数值过大
            sample_loss = (pred[i, 0] * d_map).sum() / (d_map.sum() + 1e-7)
            loss += sample_loss
        # 返回 batch 中所有样本的平均损失
        return loss / batch_size
# -----------------------------
# 定义 Dice Loss（用于语义分割）
# -----------------------------
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#
#     def forward(self, logits, targets):
#         """
#         logits: 模型输出的 logits，形状 [N, C, H, W]
#         targets: 分割标签，形状 [N, H, W]，类型为 LongTensor，取值范围为 {0,1,...,C-1}
#         """
#         num_classes = logits.size(1)
#         # 先将 logits 经过 softmax 转换为概率，形状 [N, C, H, W]
#         probs = F.softmax(logits, dim=1)
#         # 将 targets 转为 one-hot 编码，形状转换为 [N, C, H, W]
#         targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
#
#         # 将概率和 one-hot 进行展平，形状变为 [N, C, H*W]
#         probs_flat = probs.contiguous().view(probs.size(0), num_classes, -1)
#         targets_flat = targets_onehot.contiguous().view(targets_onehot.size(0), num_classes, -1)
#
#         # 计算交集和和（加上平滑项防止除零）
#         intersection = (probs_flat * targets_flat).sum(-1)
#         union = probs_flat.sum(-1) + targets_flat.sum(-1)
#         dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
#
#         # Dice Loss 为 1 - dice_score 的平均值
#         dice_loss = 1 - dice_score.mean()
#         return dice_loss


# ----------------------------------
# 定义 Focal Loss（用于边缘监督）
# ----------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: 模型边缘输出，经过 Sigmoid 激活后的概率图，形状 [N, 1, H, W]，值在 [0,1]
        targets: 边缘标签，形状 [N, 1, H, W]，值应为0或1
        """
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # 计算正确预测的概率：当 targets 为 1 时就是 inputs，否则为 (1 - inputs)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class NormalizedChamferLoss(nn.Module):
    """
    基于 Chamfer 距离的损失函数（归一化版本），适用于二值边缘图像。

    对于预测图 pred 与目标图 target：
      1. 提取二值图中边缘点的坐标（像素值 > threshold）。
      2. 将坐标归一化到 [0, 1] 范围内（分别除以图像高度和宽度）。
      3. 分别计算预测边缘点到目标边缘点集合中最近点的距离（以及反向）。
      4. 最终损失为两部分距离均值之和，再对批次取平均。

    参数：
        threshold (float): 二值化时的阈值（默认 0.5）。
        squared (bool): 是否使用平方距离（默认 False）。
    """

    def __init__(self, threshold=0.5, squared=False):
        super(NormalizedChamferLoss, self).__init__()
        self.threshold = threshold
        self.squared = squared

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 预测边缘图，形状为 (B, H, W) 或 (B, 1, H, W)。
            target (Tensor): 目标边缘图，形状同上。
        Returns:
            loss (Tensor): 标量损失。
        """
        # 如果有 channel 维度则去掉
        if pred.dim() == 4:
            pred = pred.squeeze(1)
            target = target.squeeze(1)

        batch_size, H, W = pred.shape
        # 确保loss为 tensor 类型，方便后续反向传播
        loss = torch.tensor(0.0, device=pred.device)

        for i in range(batch_size):
            pred_img = pred[i]
            target_img = target[i]

            # 提取边缘点的坐标，得到的坐标格式为 [row, col]
            pred_coords = torch.nonzero(pred_img > self.threshold, as_tuple=False).float()
            target_coords = torch.nonzero(target_img > self.threshold, as_tuple=False).float()

            # 如果某张图没有边缘点，直接跳过
            if pred_coords.shape[0] == 0 or target_coords.shape[0] == 0:
                continue

            # 将坐标归一化到 [0,1]
            # 注意：归一化时使用 (H-1) 和 (W-1) 避免最高坐标变为 1 以外的值
            pred_coords[:, 0] = pred_coords[:, 0] / (H - 1)
            pred_coords[:, 1] = pred_coords[:, 1] / (W - 1)
            target_coords[:, 0] = target_coords[:, 0] / (H - 1)
            target_coords[:, 1] = target_coords[:, 1] / (W - 1)

            # 计算两点集合之间的欧氏距离矩阵
            # pred_coords shape: (N, 2)，target_coords shape: (M, 2)
            # 利用广播扩展，diff 的 shape 为 (N, M, 2)
            diff = pred_coords.unsqueeze(1) - target_coords.unsqueeze(0)
            dist = torch.norm(diff, dim=2)  # 得到 (N, M) 的距离矩阵

            if self.squared:
                dist = dist ** 2

            # 对于预测中的每个点，找到其在目标集合中最近的距离
            min_dist_pred, _ = torch.min(dist, dim=1)
            # 对于目标中的每个点，找到其在预测集合中最近的距离
            min_dist_target, _ = torch.min(dist, dim=0)

            # 当前样本的 Chamfer 损失：两部分距离的均值之和
            sample_loss = torch.mean(min_dist_pred) + torch.mean(min_dist_target)
            loss = loss + sample_loss

        loss = loss / batch_size
        return loss


# 示例：如何使用归一化的 Chamfer 损失函数
if __name__ == "__main__":
    batch_size, H, W = 2, 64, 64
    pred = torch.zeros(batch_size, H, W)
    target = torch.zeros(batch_size, H, W)

    # 随机在每张图中设置 10 个边缘点
    for i in range(batch_size):
        idx = torch.randint(0, H * W, (10,))
        pred.view(batch_size, -1)[i, idx] = 1.0

        idx = torch.randint(0, H * W, (10,))
        target.view(batch_size, -1)[i, idx] = 1.0

    loss_fn = NormalizedChamferLoss(threshold=0.5, squared=False)
    loss_value = loss_fn(pred, target)
    print("Normalized Chamfer Loss:", loss_value.item())