import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np


from utils import rgb_to_grayscale, visualize_images, generate_edge_label
# 导入你三个模型（确保你的models文件夹中有对应模块）
from models.Unet import Unet
from models.EdgeUnet import EdgeUnet
from models.UnetPlusPlus import UnetPlusPlus

from models.Loss import BoundaryLoss, NormalizedChamferLoss

# 导入你的数据集和数据增强（请根据实际情况调整）
from data.ToothDataset import ToothDataset
# from augmentation.AddCannyEdge import AddCannyEdge  # 如有需要




def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, save_dir='checkpoints', log_dir='runs', lambda_edge=1):
    # 如果保存权重的文件夹不存在，则创建
    save_dir = os.path.join(save_dir, model.name)
    os.makedirs(save_dir, exist_ok=True)

    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # 定义损失函数
    seg_loss_fn = nn.BCEWithLogitsLoss()
    edge_loss_fn = NormalizedChamferLoss()


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_seg_loss = 0.0
        running_edge_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()



            if model.name == "edgeunet":
                # 生成边缘标签
                seg_out, edge_out = model(inputs)

                seg_loss = seg_loss_fn(seg_out, labels)

                # 取input的灰度，来计算边缘
                # edge_gt = generate_edge_label(rgb_to_grayscale(inputs).cpu().numpy())
                edge_gt = generate_edge_label(labels.cpu().numpy())
                edge_loss = edge_loss_fn(edge_out, edge_gt.to(device).float())

                # visualize_images(np.array(edge_gt.cpu()), "Ground Truth Images")
                # visualize_images(edge_out.detach().cpu().numpy(), "Output Images")
                # if epoch % 20 == 0:
                #     visualize_images(edge_gt.cpu().numpy(), "Ground Truth Images")
                #     visualize_images(edge_out.detach().cpu().numpy(), "Output Images")

                running_seg_loss += seg_loss.item()
                running_edge_loss += edge_loss.item()

                loss = seg_loss + lambda_edge * edge_loss

            elif model.name == "unet":
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            elif model.name == "unet++":
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            else:
                raise ValueError("Unsupported model type")

            loss.backward()
            optimizer.step()
            running_loss += loss.item()



        epoch_train_loss = running_loss / len(train_loader)
        epoch_seg_loss = running_seg_loss / len(train_loader)
        epoch_edge_loss = running_edge_loss / len(train_loader)
        writer.add_scalar('Loss/train', epoch_train_loss, epoch + 1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if model.name == "edgeunet":
                    # 生成边缘标签
                    seg_out, edge_out = model(inputs)
                    seg_loss = seg_loss_fn(seg_out, labels)
                    # edge_gt = generate_edge_label(rgb_to_grayscale(inputs).cpu().numpy())
                    edge_gt = generate_edge_label(labels.cpu().numpy())
                    edge_loss = edge_loss_fn(edge_out, edge_gt.to(device).float())
                    running_seg_loss += seg_loss.item()
                    running_edge_loss += edge_loss.item()
                    loss = seg_loss + lambda_edge * edge_loss

                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
        epoch_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}]\t'
              f'Train Loss: {epoch_train_loss:.4f}\t'
              f'Val Loss: {epoch_val_loss:.4f}\t'
              f'SegLoss: {epoch_seg_loss:.4f}\t'
              f'EdgeLoss: {epoch_edge_loss:.4f}\t'
              f'lr: {optimizer.param_groups[0]["lr"]}')
        writer.add_scalar('Loss/val', epoch_val_loss, epoch + 1)

        # 每隔一定epoch保存权重
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'{model.name}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train segmentation model with edge supervision")
    parser.add_argument("--model_type", type=str, default="unet++", choices=["unet", "unet++", "edgeunet"],
                        help="选择模型类型")
    parser.add_argument("--data_dir", type=str, default="F:/Datasets/tooth/tooth_seg_new_split_data", help="数据集目录")
    parser.add_argument("--split", type=str, default="train", help="数据集划分，比如 train 或 test")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存权重的目录")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard 日志目录")
    parser.add_argument("--lambda_edge", type=float, default=1, help="边缘损失权重")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理（如需使用边缘增强，可替换为对应变换）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 创建数据集
    full_dataset = ToothDataset(data_dir=args.data_dir, split=args.split, transform=transform)
    total_length = len(full_dataset)
    train_length = int(total_length * 0.9)
    val_length = total_length - train_length
    train_dataset, val_dataset = random_split(full_dataset, [train_length, val_length])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 根据命令行选择模型
    if args.model_type == "unet":
        model = Unet(in_channels=3, out_channels=1)
    elif args.model_type == "edgeunet":
        model = EdgeUnet(in_channels=3, out_channels=1)
    elif args.model_type == "unet++":
        model = UnetPlusPlus(in_channels=3, out_channels=1)
    else:
        raise ValueError("Unsupported model type")

    model.to(device)

    # checkpoint_path = "checkpoints/edgeunet_epoch_100_0.2_0.8.pth"
    # if os.path.exists(checkpoint_path):
    #     print(f"Loading pretrained weights from {checkpoint_path}")
    #     model.load_state_dict(torch.load(checkpoint_path))


    # 损失函数与优化器
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train(model, train_loader, val_loader, criterion, optimizer, device,
          num_epochs=args.epochs, save_dir=args.save_dir, log_dir=args.log_dir, lambda_edge=args.lambda_edge)
