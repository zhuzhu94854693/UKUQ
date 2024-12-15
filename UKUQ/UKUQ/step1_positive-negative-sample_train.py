import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

import numpy as np


class ChangeDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files_A = sorted(os.listdir(os.path.join(data_dir, 'A')))
        self.image_files_B = sorted(os.listdir(os.path.join(data_dir, 'B')))
        self.image_files_label = sorted(os.listdir(os.path.join(data_dir, 'label')))
        self.transform = transform

    def __getitem__(self, index):
        filename_A = os.path.join(self.data_dir, 'A', self.image_files_A[index])
        filename_B = os.path.join(self.data_dir, 'B', self.image_files_B[index])
        filename_path = os.path.join(self.data_dir, 'label', self.image_files_label[index])

        image_A = Image.open(filename_A).convert("RGB")
        image_B = Image.open(filename_B).convert("RGB")
        label = Image.open(filename_path).convert("RGB")

        if self.transform is not None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            label = self.transform(label)

        return image_A, image_B, label

    def __len__(self):
       return len(self.image_files_A)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

# 创建数据集实例
data_dir = "/home/user/zly/data2/WHU_clip/sample/orgion/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

# 创建输出目录
output_dir_0 = "/home/user/zly/daima/train_clip_0"  # labels为全0的图像对应的images_A、images_B、labels保存的目录
output_dir_1 = "/home/user/zly/daima/train_clip_1"  # labels不全为0的图像对应的images_A、images_B、labels保存的目录
os.makedirs(output_dir_0, exist_ok=True)
os.makedirs(output_dir_1, exist_ok=True)


for i, (images_A, images_B, labels) in enumerate(dataloader):
    images_A = torch.stack(images_A)
    images_B = torch.stack(images_B)
    labels = torch.stack(labels)

    images_A = images_A.squeeze(0)
    images_B = images_B.squeeze(0)
    labels = labels.squeeze(0)

    # 判断label是否全为0
    if torch.sum(labels) == 0:
        # 保存到输出目录1
        output_folder = output_dir_0
    else:
        # 保存到输出目录2
        output_folder = output_dir_1

    # 原始文件名
    filename_A = dataset.image_files_A[i]
    filename_B = dataset.image_files_B[i]
    filename_label = dataset.image_files_label[i]


    # 构建输出文件路径
    output_file_A = os.path.join(output_folder+'/A', filename_A)
    output_file_B = os.path.join(output_folder+'/B', filename_B)
    output_file_label = os.path.join(output_folder+'/label', filename_label)

    # 将张量转换为NumPy数组，并将其转换为图像对象
    image_A = Image.fromarray(np.uint8(images_A.permute(1, 2, 0).numpy() * 255))
    image_B = Image.fromarray(np.uint8(images_B.permute(1, 2, 0).numpy() * 255))
    label = Image.fromarray(np.uint8(labels.permute(1, 2, 0).numpy() * 255))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)






