import os
from datetime import datetime
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.nn.utils.prune as prune

from utils.tools import *
from models.EdgeUnet import EdgeUnet
from utils.Loss import *
from data.ToothDataset import ToothDataset
from evaluate import evaluate


def calculate_loss(model, criterion, inputs, labels):
    if model.name == "edgeunet":
        # 生成边缘标签
        seg_out, edge_outs = model(inputs)
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
    else:
        raise ValueError("Unsupported model type")
    return loss

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


def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=10,
          save_dir='checkpoints', log_dir='runs', lambda_edge=1):
    # 如果保存权重的文件夹不存在，则创建
    save_dir = os.path.join(save_dir, model.name)
    os.makedirs(save_dir, exist_ok=True)

    # 创建TensorBoard的SummaryWriter, 用日期和时间命名
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, model.name, current_date)
    writer = SummaryWriter(log_dir=log_dir)

    max_pixel_accuracy = 0.0
    max_precision = 0.0
    min_recall = 1.0
    max_f1_score = 0.0
    max_iou = 0.0
    max_dice = 0.0
    max_miou = 0.0
    max_boundary_f1 = 0.0
    min_val_loss = 999.0

    for epoch in range(num_epochs):
        model.train()

        # 训练
        epoch_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # scheduler.step()
        writer.add_scalar('Loss/train', epoch_train_loss, epoch + 1)

        # 验证
        epoch_val_loss = val_one_epoch(model, val_loader, criterion, device)

        pixel_accuracy, precision, recall, f1_score, iou, dice, miou, boundary_f1 = evaluate(model, test_loader, device)

        # Log the metrics to TensorBoard
        writer.add_scalar('Metrics/Pixel Accuracy', pixel_accuracy, epoch + 1)
        writer.add_scalar('Metrics/Precision', precision, epoch + 1)
        writer.add_scalar('Metrics/Recall', recall, epoch + 1)
        writer.add_scalar('Metrics/F1 Score', f1_score, epoch + 1)
        writer.add_scalar('Metrics/IoU', iou, epoch + 1)
        writer.add_scalar('Metrics/Dice', dice, epoch + 1)
        writer.add_scalar('Metrics/Mean IoU', miou, epoch + 1)
        writer.add_scalar('Metrics/Boundary F1', boundary_f1, epoch + 1)

        # 保存最佳权重
        if pixel_accuracy > max_pixel_accuracy:
            max_pixel_accuracy = pixel_accuracy
            save_path = os.path.join(save_dir, f'{model.name}_best_pixel_accuracy.pth')
            torch.save(model.state_dict(), save_path)
        if precision > max_precision:
            max_precision = precision
            save_path = os.path.join(save_dir, f'{model.name}_best_precision.pth')
            torch.save(model.state_dict(), save_path)
        if recall < min_recall:
            min_recall = recall
            save_path = os.path.join(save_dir, f'{model.name}_best_recall.pth')
            torch.save(model.state_dict(), save_path)
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            save_path = os.path.join(save_dir, f'{model.name}_best_f1_score.pth')
            torch.save(model.state_dict(), save_path)
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
              f'Pixel Accuracy: {pixel_accuracy:.4f}\t'
              f'Precision: {precision:.4f}\t'
              f'Recall: {recall:.4f}\t'
              f'F1 Score: {f1_score:.4f}\t'
              f'IoU: {iou:.4f}\t'
              f'Dice: {dice:.4f}\t'
              f'Mean IoU: {miou:.4f}\t'
              f'Boundary F1: {boundary_f1:.4f}\t')
        writer.add_scalar('Loss/val', epoch_val_loss, epoch + 1)

        # 每隔一定epoch保存权重
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'{model.name}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)

    print("Training complete.")
    print("Best Metrics:" 
          f"Pixel Accuracy: {max_pixel_accuracy:.4f}\t"
          f"Precision: {max_precision:.4f}\t"
          f"Recall: {min_recall:.4f}\t"
          f"F1 Score: {max_f1_score:.4f}\t"
          f"IoU: {max_iou:.4f}\t"
          f"Dice: {max_dice:.4f}\t"
          f"Mean IoU: {max_miou:.4f}\t"
          f"Boundary F1: {max_boundary_f1:.4f}\t")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train segmentation model with edge supervision")
    parser.add_argument("--model_type", type=str, default="edgeunet",
                        choices=["unet", "edgeunet", "egeunet"],
                        help="选择模型类型")
    parser.add_argument("--data_dir", type=str, default="F:/Datasets/tooth/tooth_seg_new_split_data",
                        help="数据集目录")
    parser.add_argument("--split", type=str, default="train", help="数据集划分，比如 train 或 test")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存权重的目录")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard 日志目录")
    parser.add_argument("--lambda_edge", type=float, default=1, help="边缘损失权重")
    parser.add_argument("--prune_amount", type=float, default=0.3, help="剪枝比例")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
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

    test_dataset = ToothDataset(data_dir=args.data_dir, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    if args.model_type == "edgeunet":
        model = EdgeUnet(in_channels=3, out_channels=1)
    else:
        raise ValueError("Only EdgeUnet is supported for pruning in this example")

    model.to(device)
    summary(model, (3, 256, 256))

    # 损失函数与优化器
    criterion = edge_bacediceloss(wb=1, wd=1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 训练原始模型
    print("Training original model...")
    # train(model, train_loader, val_loader, test_loader, criterion, optimizer, device,
    #       num_epochs=args.epochs, save_dir=args.save_dir, log_dir=args.log_dir, lambda_edge=args.lambda_edge)
    checkpoint_path = "checkpoints/edgeunet/edgeunet_epoch_100.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))


    # 评估原始模型
    print("\nEvaluating original model...")
    model.eval()
    orig_metrics = evaluate(model, test_loader, device)
    print("Original Model Metrics:")
    print(f"Pixel Accuracy: {orig_metrics[0]:.4f}, Precision: {orig_metrics[1]:.4f}, Recall: {orig_metrics[2]:.4f}, "
          f"F1 Score: {orig_metrics[3]:.4f}, IoU: {orig_metrics[4]:.4f}, Dice: {orig_metrics[5]:.4f}, "
          f"Mean IoU: {orig_metrics[6]:.4f}, Boundary F1: {orig_metrics[7]:.4f}")

    # 剪枝模型
    print("\nApplying structured pruning...")
    parameters_to_prune = [(module, 'weight') for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    for module, param_name in parameters_to_prune:
        prune.ln_structured(module, name=param_name, amount=args.prune_amount, n=2, dim=0)  # 按通道剪枝

    # 检查剪枝后的稀疏性
    print("Sparsity after pruning:")
    for name, (module, _) in enumerate(parameters_to_prune):
        sparsity = (getattr(module, 'weight') == 0).float().mean().item()
        print(f"Module {name} (Conv2d): sparsity = {sparsity:.2%}")

    # 永久移除剪枝权重
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    # 评估剪枝后的模型
    print("\nEvaluating pruned model...")
    model.eval()
    pruned_metrics = evaluate(model, test_loader, device)
    print("Pruned Model Metrics:")
    print(f"Pixel Accuracy: {pruned_metrics[0]:.4f}, Precision: {pruned_metrics[1]:.4f}, Recall: {pruned_metrics[2]:.4f}, "
          f"F1 Score: {pruned_metrics[3]:.4f}, IoU: {pruned_metrics[4]:.4f}, Dice: {pruned_metrics[5]:.4f}, "
          f"Mean IoU: {pruned_metrics[6]:.4f}, Boundary F1: {pruned_metrics[7]:.4f}")

    # 可选：微调剪枝后的模型
    print("\nFine-tuning pruned model...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr / 10)  # 降低学习率
    train(model, train_loader, val_loader, test_loader, criterion, optimizer, device,
          num_epochs=5, save_dir=args.save_dir, log_dir=args.log_dir + "_finetune", lambda_edge=args.lambda_edge)

    # 评估微调后的模型
    print("\nEvaluating fine-tuned pruned model...")
    model.eval()
    finetuned_metrics = evaluate(model, test_loader, device)
    print("Fine-tuned Pruned Model Metrics:")
    print(f"Pixel Accuracy: {finetuned_metrics[0]:.4f}, Precision: {finetuned_metrics[1]:.4f}, Recall: {finetuned_metrics[2]:.4f}, "
          f"F1 Score: {finetuned_metrics[3]:.4f}, IoU: {finetuned_metrics[4]:.4f}, Dice: {finetuned_metrics[5]:.4f}, "
          f"Mean IoU: {finetuned_metrics[6]:.4f}, Boundary F1: {finetuned_metrics[7]:.4f}")

    # 比较结果
    print("\nComparison of Metrics:")
    print(f"{'Metric':<15} {'Original':<12} {'Pruned':<12} {'Fine-tuned':<12}")
    print(f"{'Pixel Acc':<15} {orig_metrics[0]:<12.4f} {pruned_metrics[0]:<12.4f} {finetuned_metrics[0]:<12.4f}")
    print(f"{'Precision':<15} {orig_metrics[1]:<12.4f} {pruned_metrics[1]:<12.4f} {finetuned_metrics[1]:<12.4f}")
    print(f"{'Recall':<15} {orig_metrics[2]:<12.4f} {pruned_metrics[2]:<12.4f} {finetuned_metrics[2]:<12.4f}")
    print(f"{'F1 Score':<15} {orig_metrics[3]:<12.4f} {pruned_metrics[3]:<12.4f} {finetuned_metrics[3]:<12.4f}")
    print(f"{'IoU':<15} {orig_metrics[4]:<12.4f} {pruned_metrics[4]:<12.4f} {finetuned_metrics[4]:<12.4f}")
    print(f"{'Dice':<15} {orig_metrics[5]:<12.4f} {pruned_metrics[5]:<12.4f} {finetuned_metrics[5]:<12.4f}")
    print(f"{'Mean IoU':<15} {orig_metrics[6]:<12.4f} {pruned_metrics[6]:<12.4f} {finetuned_metrics[6]:<12.4f}")
    print(f"{'Boundary F1':<15} {orig_metrics[7]:<12.4f} {pruned_metrics[7]:<12.4f} {finetuned_metrics[7]:<12.4f}")

    # 保存微调后的模型
    save_path = os.path.join(args.save_dir, f'{model.name}_pruned_finetuned.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned pruned model saved to {save_path}")
