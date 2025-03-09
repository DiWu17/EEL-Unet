import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from augmentation.AddCannyEdge import AddCannyEdge
from augmentation.CannyEnhance import CannyEnhance

class ToothDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        初始化数据集
        :param data_dir: 数据集所在的根目录
        :param split: 选择使用的分割，'train' 或 'test'
        :param transform: 数据预处理的变换操作
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # 构造图片和标签的路径
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'masks')

        # 获取所有图像文件名
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        # 确保图像和标签数量一致
        assert len(self.image_files) == len(self.mask_files), "Image and mask counts do not match!"

    def __len__(self):
        """返回数据集的大小"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: 数据索引
        :return: 图像和标签
        """
        # 加载图像和标签
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert('RGB')  # 将图像转换为RGB格式

        mask = Image.open(mask_path).convert('L')  # 将mask图像转换为灰度图

        # 添加 Canny 边缘通道
        # image = AddCannyEdge()(image)

        # 添加 Canny 边缘增强
        # image = CannyEnhance(low_threshold=100, high_threshold=200, edge_color=(255, 255, 255), alpha=0.2)(image)

        # 应用数据预处理变换（如果有）
        if self.transform:
            image = self.transform(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            mask = self.transform(mask)

        return image, mask


if __name__ == '__main__':

    # 数据预处理：示例变换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # 使用示例
    train_dataset = ToothDataset(data_dir='F:/Datasets/tooth/tooth_seg_new_split_data', split='train', transform=transform)
    test_dataset = ToothDataset(data_dir='F:/Datasets/tooth/tooth_seg_new_split_data', split='test', transform=transform)

    # 可以使用 DataLoader 来加载数据
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
