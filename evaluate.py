import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2  # 用于形态学操作
from PIL import Image
import argparse
from datetime import datetime

# 导入模型（确保 models 文件夹中有对应模块）
from models.Unet import Unet
from models.EdgeUnet import EdgeUnet
from models.UnetPlusPlus import UnetPlusPlus
from models.egeunet import EGEUNet

# 导入数据集
from data.ToothDataset import ToothDataset

def seg2bnd(mask, dilation_ratio=0.02):
    """
    从二值分割 mask 中提取边界。
    :param mask: numpy 数组，尺寸为 (H, W)，取值为 0 或 1
    :param dilation_ratio: 膨胀比例，用于确定腐蚀迭代次数
    :return: 二值边界 mask，尺寸为 (H, W)，True 表示边界像素
    """
    h, w = mask.shape
    # 根据图像尺寸和 dilation_ratio 确定腐蚀迭代次数
    dilation = int(round(np.mean([h, w]) * dilation_ratio))
    dilation = max(dilation, 1)
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=dilation)
    boundary = mask_uint8 - eroded
    boundary = boundary > 0
    return boundary

def boundary_f1_score(gt, pred, dilation_ratio=0.02):
    """
    计算单个样本的 Boundary F1 Score。
    :param gt: 真实 mask，numpy 数组，尺寸为 (H, W)，值为 0 或 1
    :param pred: 预测 mask，numpy 数组，尺寸为 (H, W)，值为 0 或 1
    :param dilation_ratio: 用于边界提取的膨胀比例
    :return: Boundary F1 Score（0~1）
    """
    gt_b = seg2bnd(gt, dilation_ratio)
    pred_b = seg2bnd(pred, dilation_ratio)
    tp = np.logical_and(pred_b, gt_b).sum()
    precision = tp / (pred_b.sum() + 1e-7)
    recall = tp / (gt_b.sum() + 1e-7)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1

def evaluate(model, dataloader, device):
    """
    计算模型在测试集上的各项指标：
    像素准确率、精确率、召回率、F1 Score、IoU、Dice、平均 IoU 以及 Boundary F1 Score。
    """
    model.eval()

    TP = 0  # 真阳性
    TN = 0  # 真阴性
    FP = 0  # 假阳性
    FN = 0  # 假阴性

    boundary_f1_total = 0.0  # 累计边界 F1 Score
    sample_count = 0       # 样本数量

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # 假设 labels 尺寸为 [B, 1, H, W]，值为 0 或 1

            outputs = model(inputs)
            # 如果模型返回多个输出（例如 EdgeUNet 返回 (seg_out, edge_out)），则取第一个作为分割预测
            if model.name == "edgeunet":
                seg_out = outputs[0]
            elif model.name == "egeunet":
                seg_out = outputs[1]
            elif model.name == "eelunet":
                seg_out = outputs[1]
            else:
                seg_out = outputs

            # 将 logits 转换为概率，再二值化（阈值 0.5）
            preds = (seg_out > 0.5).float()  # 形状为 [B, 1, H, W]

            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)

            TP += ((preds_flat == 1) & (labels_flat == 1)).sum().item()
            TN += ((preds_flat == 0) & (labels_flat == 0)).sum().item()
            FP += ((preds_flat == 1) & (labels_flat == 0)).sum().item()
            FN += ((preds_flat == 0) & (labels_flat == 1)).sum().item()

            # 计算每个样本的 Boundary F1 Score
            batch_size = preds.size(0)
            for i in range(batch_size):
                pred_mask_np = preds[i, 0].cpu().numpy()
                label_np = labels[i, 0].cpu().numpy()
                boundary_f1 = boundary_f1_score(label_np, pred_mask_np)
                boundary_f1_total += boundary_f1
                sample_count += 1

    epsilon = 1e-7
    pixel_accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    iou = TP / (TP + FP + FN + epsilon)
    dice = 2 * TP / (2 * TP + FP + FN + epsilon)
    # 简单计算背景 IoU
    iou_bg = TN / (TN + FP + FN + epsilon)
    miou = (iou + iou_bg) / 2
    avg_boundary_f1 = boundary_f1_total / (sample_count + epsilon)

    return pixel_accuracy, precision, recall, f1_score, iou, dice, miou, avg_boundary_f1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate segmentation model and output metrics")
    parser.add_argument("--model_type", type=str, default="edgeunet", choices=["unet", "unet++", "edgeunet", "egeunet"],
                        help="选择模型类型")
    parser.add_argument("--data_dir", type=str, default="F:/Datasets/tooth/tooth_seg_new_split_data",
                        help="数据集目录")
    parser.add_argument("--split", type=str, default="test", help="test")
    parser.add_argument("--batch_size", type=int, default=8, help="测试时的批大小")
    parser.add_argument("--checkpoint", type=str, default="D:/python/Unet-baseline/checkpoints/edgeunet/edgeunet_best.pth", help="模型权重文件路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = ToothDataset(data_dir=args.data_dir, split=args.split, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model_type == "unet":
        model = Unet(in_channels=3, out_channels=1)
    elif args.model_type == "edgeunet":
        model = EdgeUnet(in_channels=3, out_channels=1)
    elif args.model_type == "unet++":
        model = UnetPlusPlus(in_channels=3, out_channels=1)
    elif args.model_type == "egeunet":
        model = EGEUNet(num_classes=1,
                        input_channels=3,
                        c_list=[8, 16, 24, 32, 48, 64],
                        bridge=True,
                        gt_ds=True,
                        )
    else:
        raise ValueError("Unsupported model type")
    model.to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model weights from {args.checkpoint}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}. Evaluating untrained model.")

    # 从权重文件名中提取模型名称和 epoch 信息
    base_name = os.path.basename(args.checkpoint)
    m = re.search(r'^(.*)_epoch_(\d+)', base_name)
    if m:
        model_name = m.group(1)
        epoch_str = m.group(2)
    else:
        model_name = args.model_type
        epoch_str = "unknown"

    # 获取当前日期和时间
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Evaluation date: {current_date}")
    print(f"Model: {model_name}, Epoch: {epoch_str}")
    pixel_accuracy, precision, recall, f1_score, iou, dice, miou, boundary_f1 = evaluate(model, test_loader, device)

    print("Evaluation Metrics:")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"IoU (foreground): {iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Boundary F1 Score: {boundary_f1:.4f}")

