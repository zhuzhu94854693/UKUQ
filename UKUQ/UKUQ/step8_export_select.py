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
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.10)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.10_0.10/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.20)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.20_0.20/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.30)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.30_0.30/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.40)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.40_0.40/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.50)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.50_0.50/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)




# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.60)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.60_0.60/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)




# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.70)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.70_0.70/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.80)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.80_0.80/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)


# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_0.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.90)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.90_0.90/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)


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
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.10)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.10_0.10/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.20)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.20_0.20/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.30)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.30_0.30/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.40)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.40_0.40/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.50)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.50_0.50/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)




# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.60)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.60_0.60/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)




# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.70)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.70_0.70/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)



# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.80)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.80_0.80/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)


# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'/home/user/zly/daima/UKUQ/result/result_jiejin_1.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

num_samples = len(index_list)
num_selected = math.ceil(num_samples * 0.90)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"/home/user/zly/daima/UKUQ/result/select/t0.90_0.90/train"  # 修复文件夹路径错误
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index in selected_indices:
    filename_A = dataset.image_files_A[index]
    filename_B = dataset.image_files_B[index]
    filename_label = dataset.image_files_label[index]

    output_file_A = os.path.join(output_folder, 'A', filename_A)
    output_file_B = os.path.join(output_folder, 'B', filename_B)
    output_file_label = os.path.join(output_folder, 'label', filename_label)

    image_A = Image.open(os.path.join(data_dir, 'A', filename_A))
    image_B = Image.open(os.path.join(data_dir, 'B', filename_B))
    label = Image.open(os.path.join(data_dir, 'label', filename_label))

    # 保存图像
    image_A.save(output_file_A)
    image_B.save(output_file_B)
    label.save(output_file_label)












