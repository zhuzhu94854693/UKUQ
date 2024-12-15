# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np

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



#变化的区域越小越困难，越容易检测成未变化  从小到大

result_cha_entr_AB_1 = []


Image.MAX_IMAGE_PIXELS = None

# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

for images_A, images_B, labels in dataloader:
    images_A = torch.stack(images_A)
    images_B = torch.stack(images_B)
    labels = torch.stack(labels)
    images_A = images_A.squeeze(0).numpy()
    images_B = images_B.squeeze(0).numpy()


    labels_2d = labels.squeeze(0)[0,:,:]
    labels = labels.squeeze(0).numpy()




    # 计算图像熵
    entropy_A = []
    for channel in images_A:
        hist, _ = np.histogram(channel.ravel(), bins=256, range=[0, 256])
        hist = hist / float(channel.size)
        entropy_A.append(-np.sum(hist * np.log2(hist + 1e-10)))
    # print('图像熵为：', entropy_A)
    entropy_B = []
    for channel in images_B:
        hist, _ = np.histogram(channel.ravel(), bins=256, range=[0, 256])
        hist = hist / float(channel.size)
        entropy_B.append(-np.sum(hist * np.log2(hist + 1e-10)))
    # print('图像熵为：', entropy_B)


    cha_entr_AB = [abs(a - b) for a, b in zip(entropy_A, entropy_B)]

    cha_entr_AB_1 = (cha_entr_AB[0]+cha_entr_AB[1]+cha_entr_AB[2])/3


    result_cha_entr_AB_1.append(cha_entr_AB_1)



        # 将结果从大到小排序并保存到txt文件
    sorted_results = sorted(enumerate(result_cha_entr_AB_1), key=lambda x: x[1], reverse=True)
    with open(r"/home/user/zly/daima/UKUQ/result/result_cha_entr_AB_1_max_min.txt", "w") as f:
        for index, result in sorted_results:
            f.write(f"Image {index}: cha_entr_AB_1 = {result}\n")







result_cha_entr_AB_1 = []





# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

for images_A, images_B, labels in dataloader:
    images_A = torch.stack(images_A)
    images_B = torch.stack(images_B)
    labels = torch.stack(labels)
    images_A = images_A.squeeze(0).numpy()
    images_B = images_B.squeeze(0).numpy()


    labels_2d = labels.squeeze(0)[0,:,:]
    labels = labels.squeeze(0).numpy()




    # 计算图像熵
    entropy_A = []
    for channel in images_A:
        hist, _ = np.histogram(channel.ravel(), bins=256, range=[0, 256])
        hist = hist / float(channel.size)
        entropy_A.append(-np.sum(hist * np.log2(hist + 1e-10)))
    # print('图像熵为：', entropy_A)
    entropy_B = []
    for channel in images_B:
        hist, _ = np.histogram(channel.ravel(), bins=256, range=[0, 256])
        hist = hist / float(channel.size)
        entropy_B.append(-np.sum(hist * np.log2(hist + 1e-10)))
    # print('图像熵为：', entropy_B)




    cha_entr_AB = [abs(a - b) for a, b in zip(entropy_A, entropy_B)]


    cha_entr_AB_1 = (cha_entr_AB[0]+cha_entr_AB[1]+cha_entr_AB[2])/3

    result_cha_entr_AB_1.append(cha_entr_AB_1)


        # 将结果从小到大排序并保存到txt文件
    sorted_results = sorted(enumerate(result_cha_entr_AB_1), key=lambda x: x[1], reverse=True)
    with open(r"/home/user/zly/daima/UKUQ/result/result_cha_entr_AB_0_max_min.txt", "w") as f:
        for index, result in sorted_results:
            f.write(f"Image {index}: cha_entr_AB_0 = {result}\n")

end_time = time.time()

execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")