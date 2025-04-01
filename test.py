import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from datetime import datetime
import argparse

# 导入模型
from models.Unet import Unet
from models.EELUnet import EELUnet
from models.UnetPlusPlus import UnetPlusPlus
from models.egeunet import EGEUNet
from models.malunet import MALUNet
from models.unext import UNext, UNext_S

# 导入数据集
from data.ToothDataset import ToothDataset

def save_mask(tensor, save_path):
    """
    将预测得到的二值 mask 保存为图像文件
    :param tensor: torch.Tensor，尺寸为 [H, W]，值为 0 或 1
    :param save_path: 保存路径
    """
    mask_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_np, mode='L')
    mask_img.save(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test segmentation model and save predicted masks")
    parser.add_argument("--model_type", type=str, default="eelunet",
                        choices=["unet", "eelunet", "egeunet", "unext", "unext_s", "malunet"],
                        help="选择模型类型")
    parser.add_argument("--data_dir", type=str, default="F:/Datasets/tooth/tooth_seg_new_split_data", help="数据集目录")
    parser.add_argument("--checkpoint", type=str, default="D:/python/EELUnet/checkpoints/eelunet_7385.pth", help="模型权重文件路径")
    parser.add_argument("--batch_size", type=int, default=8, help="测试时的批大小")
    parser.add_argument("--save_dir", type=str, default="results", help="保存预测结果的根目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义与训练时一致的数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 创建测试数据集
    test_dataset = ToothDataset(data_dir=args.data_dir, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 根据模型类型选择模型
    if args.model_type == "unet":
        model = Unet(in_channels=3, out_channels=1)
    elif args.model_type == "eelunet":
        model = EELUnet(in_channels=3, out_channels=1)
    elif args.model_type == "unet++":
        model = UnetPlusPlus(in_channels=3, out_channels=1)
    elif args.model_type == "egeunet":
        model = EGEUNet(num_classes=1,
                        input_channels=3,
                        c_list=[8, 16, 24, 32, 48, 64],
                        bridge=True,
                        gt_ds=True,
                        )
    elif args.model_type == "unext":
        model = UNext(num_classes=1, in_channels=3)
    elif args.model_type == "unext_s":
        model = UNext_S(num_classes=1, in_channels=3)
    elif args.model_type == "malunet":
        model = MALUNet(num_classes=1, input_channels=3)
    else:
        raise ValueError("Unsupported model type")
    model.to(device)

    # 加载权重
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model weights from {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    # 从权重文件名中提取模型名称和 epoch 信息
    base_name = os.path.basename(args.checkpoint)
    m = re.search(r'^(.*)_epoch_(\d+)', base_name)
    if m:
        model_name = m.group(1)
        epoch_str = m.group(2)
    else:
        model_name = args.model_type
        epoch_str = "unknown"

    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.save_dir, f'{model_name}_{current_date}_epoch{epoch_str}')
    os.makedirs(results_dir, exist_ok=True)

    model.eval()
    global_idx = 0
    sigmoid = nn.Sigmoid()  # 添加 Sigmoid 激活函数
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            # 处理不同模型的输出
            if getattr(model, "name", None) == "eelunet":
                outputs, _ = model(inputs)
            elif getattr(model, "name", None) == "egeunet":
                _, outputs = model(inputs)
            else:
                outputs = model(inputs)

            # 将 logits 转换为概率并二值化
            # probs = sigmoid(outputs)  # [B, C, H, W]
            preds = (outputs > 0.5).float()  # [B, C, H, W]

            batch_size = preds.size(0)  # 获取当前 batch 的实际大小
            for i in range(batch_size):
                pred_mask = preds[i, 0, :, :]  # 提取单张掩码 [H, W]
                save_filename = f"pred_{global_idx}.png"  # 改进命名
                save_path = os.path.join(results_dir, save_filename)
                save_mask(pred_mask, save_path)
                print(f"Saved mask: {save_path}")
                global_idx += 1

    print(f"Testing complete. Predicted masks saved to {results_dir}")