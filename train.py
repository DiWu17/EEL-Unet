import os
from datetime import datetime
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import json

from models.Unet import Unet
from models.EELUnet import EELUnet
from models.UnetPlusPlus import UnetPlusPlus
from models.egeunet import EGEUNet
from models.unext import UNext, UNext_S
from models.malunet import MALUNet

from utils.Loss import *
from utils.tools import *

from data.ToothDataset import ToothDataset
from evaluate import evaluate

def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 10000)
        print(f"Using seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    torch.backends.cudnn.deterministic = True  # 确保卷积操作确定性
    torch.backends.cudnn.benchmark = False     # 关闭自动优化以保证可重复性

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = calculate_loss(model, criterion, inputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_train_loss = running_loss / len(train_loader)

    return epoch_train_loss


def val_one_epoch(model, val_loader, criterion, device):
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = calculate_loss(model, criterion, inputs, labels)
            val_loss += loss.item()
    epoch_val_loss = val_loss / len(val_loader)
    return epoch_val_loss


def calculate_loss(model, criterion, inputs, labels):
    if model.name == "eelunet":
        seg_out, edge_outs = model(inputs)
        # loss = criterion(seg_out, labels)
        loss = criterion(edge_outs, seg_out, labels)
    elif model.name == "unet":
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    elif model.name == "unet++":
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    elif model.name == "egeunet":
        gt_pre, out = model(inputs)
        loss = criterion(gt_pre, out, labels)
    elif model.name == "unext" or model.name == "unext_s":
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    elif model.name == "malunet":
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    else:
        raise ValueError("Unsupported model type")
    return loss


def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=10,
          save_dir='checkpoints', log_dir='runs', lambda_edge=1):
    # 如果保存权重的文件夹不存在，则创建
    save_dir = os.path.join(save_dir, model.name)
    os.makedirs(save_dir, exist_ok=True)

    # 创建TensorBoard的SummaryWriter, 用日期和时间命名
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, model.name, current_date)
    writer = SummaryWriter(log_dir=log_dir)

    # 创建用于保存loss的字典
    loss_history = {
        'train_loss': [],
        'val_loss': []
    }

    # 定义JSON文件路径
    json_path = os.path.join(save_dir, 'loss_history.json')

    max_iou = 0.0
    max_dice = 0.0
    max_miou = 0.0
    max_mdice = 0.0
    max_boundary_f1 = 0.0
    min_val_loss = 999.0

    for epoch in range(num_epochs):
        model.train()
        # 训练
        epoch_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        writer.add_scalar('Loss/train', epoch_train_loss, epoch + 1)

        # 验证
        epoch_val_loss = val_one_epoch(model, val_loader, criterion, device)

        # 将loss添加到历史记录
        loss_history['train_loss'].append(float(epoch_train_loss))  # 转换为float避免JSON序列化问题
        loss_history['val_loss'].append(float(epoch_val_loss))
        pixel_accuracy, precision, recall, f1_score, iou, dice, miou, boundary_f1, mdice = evaluate(model, test_loader,
                                                                                                    device)
        # Log the metrics to TensorBoard
        # writer.add_scalar('Metrics/Pixel Accuracy', pixel_accuracy, epoch + 1)
        # writer.add_scalar('Metrics/Precision', precision, epoch + 1)
        # writer.add_scalar('Metrics/Recall', recall, epoch + 1)
        # writer.add_scalar('Metrics/F1 Score', f1_score, epoch + 1)
        # writer.add_scalar('Metrics/IoU', iou, epoch + 1)
        writer.add_scalar('Metrics/Dice', dice, epoch + 1)
        writer.add_scalar('Metrics/Mean IoU', miou, epoch + 1)
        writer.add_scalar('Metrics/Mean Dice', mdice, epoch + 1)
        writer.add_scalar('Metrics/Boundary F1', boundary_f1, epoch + 1)

        # 保存最佳权重
        # if pixel_accuracy > max_pixel_accuracy:
        #     max_pixel_accuracy = pixel_accuracy
        #     save_path = os.path.join(save_dir, f'{model.name}_best_pixel_accuracy.pth')
        #     torch.save(model.state_dict(), save_path)
        # if precision > max_precision:
        #     max_precision = precision
        #     save_path = os.path.join(save_dir, f'{model.name}_best_precision.pth')
        #     torch.save(model.state_dict(), save_path)
        # if recall < min_recall:
        #     min_recall = recall
        #     save_path = os.path.join(save_dir, f'{model.name}_best_recall.pth')
        #     torch.save(model.state_dict(), save_path)
        # if f1_score > max_f1_score:
        #     max_f1_score = f1_score
        #     save_path = os.path.join(save_dir, f'{model.name}_best_f1_score.pth')
        #     torch.save(model.state_dict(), save_path)
        if iou > max_iou:
            max_iou = iou
            save_path = os.path.join(save_dir, f'{model.name}_best_iou.pth')
            torch.save(model.state_dict(), save_path)
        if dice > max_dice:
            max_dice = dice
            save_path = os.path.join(save_dir, f'{model.name}_best_dice.pth')
            torch.save(model.state_dict(), save_path)
        if miou > max_miou:
            max_miou = miou
            save_path = os.path.join(save_dir, f'{model.name}_best_miou.pth')
            torch.save(model.state_dict(), save_path)
        if mdice > max_mdice:
            max_mdice = mdice
            save_path = os.path.join(save_dir, f'{model.name}_best_mdice.pth')
            torch.save(model.state_dict(), save_path)
        if boundary_f1 > max_boundary_f1:
            max_boundary_f1 = boundary_f1
            save_path = os.path.join(save_dir, f'{model.name}_best_boundary_f1.pth')
            torch.save(model.state_dict(), save_path)
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            save_path = os.path.join(save_dir, f'{model.name}_best.pth')
            torch.save(model.state_dict(), save_path)

        # 打印当前 epoch 的训练和验证损失
        print(f'Epoch [{epoch + 1}/{num_epochs}]\t'
              f'Train Loss: {epoch_train_loss:.4f}\t'
              f'Val Loss: {epoch_val_loss:.4f}\t'
              f'lr: {optimizer.param_groups[0]["lr"]} ',
              f'IoU: {iou:.4f}\t'
              f'Dice: {dice:.4f}\t'
              f'Mean IoU: {miou:.4f}\t'
              f'Mean Dice: {mdice:.4f}\t'
              f'Boundary F1: {boundary_f1:.4f}\t')
        writer.add_scalar('Loss/val', epoch_val_loss, epoch + 1)

        # 每隔一定epoch保存权重
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'{model.name}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)

        # 在每个epoch结束后保存loss到JSON文件
        with open(json_path, 'w') as f:
            json.dump(loss_history, f, indent=4)

    print("Training complete.")
    print("Best Metrics:"
          f"IoU: {max_iou:.4f}\t"
          f"Dice: {max_dice:.4f}\t"
          f"Mean IoU: {max_miou:.4f}\t"
          f"Boundary F1: {max_boundary_f1:.4f}\t")

    # 训练结束后最后一次保存loss
    with open(json_path, 'w') as f:
        json.dump(loss_history, f, indent=4)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train segmentation model with edge supervision")
    parser.add_argument("--model_type", type=str, default="eelunet",
                        choices=["unet", "eelunet", "egeunet", "unext", "unext_s", "malunet"],
                        help="选择模型类型")
    # parser.add_argument("--data_dir", type=str, default="F:/Datasets/2DSegmentation/ISIC2017", help="数据集目录")
    # parser.add_argument("--data_dir", type=str, default="F:/Datasets/2DSegmentation/kvasir", help="数据集目录")
    parser.add_argument("--data_dir", type=str, default="F:/Datasets/tooth/tooth_seg_new_split_data",
                        help="数据集目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存权重的目录")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard 日志目录")
    parser.add_argument("--lambda_edge", type=float, default=1, help="边缘损失权重")
    parser.add_argument("--seed", type=int, help="随机种子")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.seed:
        # 7686 Mean IoU: 0.9606	Mean Dice: 0.9799	Boundary F1: 0.7365
        # 6716 Mean IoU: 0.9562	Mean Dice: 0.9776	Boundary F1: 0.7385
        # 6345 Mean IoU: 0.9611	Mean Dice: 0.9802	Boundary F1: 0.7400

        set_seed(args.seed)
    else:
        set_seed()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转化为Tensor
    ])

    # 创建数据集
    full_dataset = ToothDataset(data_dir=args.data_dir, split="train", transform=transform)
    total_length = len(full_dataset)
    train_length = int(total_length * 0.9)
    val_length = total_length - train_length
    train_dataset, val_dataset = random_split(full_dataset, [train_length, val_length])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = ToothDataset(data_dir=args.data_dir, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 根据命令行选择模型
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
    summary(model, (3, 256, 256))

    # 加载预训练权重
    # checkpoint_path = "checkpoints/eelunet/eelunet_epoch_50.pth"
    # if os.path.exists(checkpoint_path):
    #     print(f"Loading pretrained weights from {checkpoint_path}")
    #     model.load_state_dict(torch.load(checkpoint_path))

    # 损失函数
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = DiceLoss()
    # criterion = BceDiceLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = GT_BceDiceLoss(wb=1, wd=1)
    criterion = edge_BceDiceLoss(wb=1, wd=1)
    # criterion = edge_Diceloss()
    # criterion = edge_BceLoss()

    # 优化器
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.90, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 训练.
    train(model, train_loader, val_loader, test_loader, criterion, optimizer, device,
          num_epochs=args.epochs, save_dir=args.save_dir, log_dir=args.log_dir, lambda_edge=args.lambda_edge)
