import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


import time

start_time = time.time()

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
data_dir = r"/home/user/zly/daima/UKUQ/result/diedai/xuanqu_diedai/t0.20_0.20/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_0.20_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.10) + math.ceil(1331 * 0.10)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.10.txt', 'w') as f:


    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")



# 创建数据集实例
data_dir = r"/home/user/zly/daima/UKUQ/result/diedai/xuanqu_diedai/t0.30_0.30/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_0.30_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.20) + math.ceil(1331 * 0.20)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.20.txt', 'w') as f:


    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")


# 创建数据集实例
data_dir = r"/home/user/zly/daima/UKUQ/result/diedai/xuanqu_diedai/t0.40_0.40/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_0.40_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.30) + math.ceil(1331 * 0.30)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.30.txt', 'w') as f:


    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")

# 创建数据集实例
data_dir = r"/home/user/zly/daima/UKUQ/result/diedai/xuanqu_diedai/t0.50_0.50/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_0.50_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.40) + math.ceil(1331 * 0.40)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.40.txt', 'w') as f:


    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")


# 创建数据集实例
data_dir = r"/home/user/zly/daima/UKUQ/result/diedai/xuanqu_diedai/t0.60_0.60/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_0.60_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.50) + math.ceil(1331 * 0.50)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.50.txt', 'w') as f:


    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")




# 创建数据集实例
data_dir = r"/home/user/zly/daima/UKUQ/result/diedai/xuanqu_diedai/t0.70_0.70/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_0.70_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.60) + math.ceil(1331 * 0.60)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.60.txt', 'w') as f:


    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")





# 创建数据集实例
data_dir = r"/home/user/zly/daima/UKUQ/result/diedai/xuanqu_diedai/t0.80_0.80/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_0.80_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.70) + math.ceil(1331 * 0.70)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.70.txt', 'w') as f:


    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")





# 创建数据集实例
data_dir = r"/home/user/zly/daima/UKUQ/result/diedai/xuanqu_diedai/t0.90_0.90/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_0.90_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.80) + math.ceil(1331 * 0.80)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.80.txt', 'w') as f:


    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")



# 创建数据集实例
data_dir = r"/home/user/zly/data2/WHU_clip/sample/orgion/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_all_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 0.90) + math.ceil(1331 * 0.90)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_0.90.txt', 'w') as f:

    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")



# 创建数据集实例
data_dir = r"/home/user/zly/data2/WHU_clip/sample/orgion/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/diedai/pixel_count_train_all_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_selected = math.ceil(4003 * 1) + math.ceil(1331 * 1)
selected_indices = index_list[:num_selected]

# 写入对应的pixel_count结果到文件
with open(r'/home/user/zly/daima/UKUQ/result/diedai/diedai_train_all.txt', 'w') as f:

    for index, index_file in enumerate(selected_indices, start=1):
        filename_A = dataset.image_files_A[index_file]
        f.write(f"Image {filename_A}: pixel_count = {index}\n")



end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")








