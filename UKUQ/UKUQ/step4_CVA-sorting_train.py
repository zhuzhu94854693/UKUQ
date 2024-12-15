import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN

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
data_dir = "/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

results = []

for images_A, images_B, labels in dataloader:
    images_A = torch.stack(images_A)
    images_B = torch.stack(images_B)
    labels = torch.stack(labels)

    #traditional meothod
    epsilon = 1e-8
    dist = (torch.abs(images_A.squeeze(0) - images_B.squeeze(0)) + epsilon) * (
            torch.abs(images_A.squeeze(0) - images_B.squeeze(0)) + epsilon)

    su = torch.sum(dist, dim=0)
    ed_img00_feat_list = torch.sqrt(su + epsilon)
    ed_img00_feat_list = (ed_img00_feat_list - torch.min(ed_img00_feat_list)) / (
            torch.max(ed_img00_feat_list) - torch.min(ed_img00_feat_list) + epsilon)



    d = images_A.squeeze(0) * images_B.squeeze(0)
    d1 = images_A.squeeze(0) * images_A.squeeze(0)
    d2 = images_B.squeeze(0) * images_B.squeeze(0)


    t = torch.sum(d, dim=0)
    t1 = torch.sum(d1, dim=0)
    t2 = torch.sum(d2, dim=0)

    sam_img00_feat_list = (t / torch.clamp((t1 * t2 + epsilon).sqrt(), min=1e-6, max=float('inf')))
    sam_img00_feat_list = torch.max(sam_img00_feat_list)-sam_img00_feat_list
    sam_img00_feat_list = (sam_img00_feat_list - torch.min(sam_img00_feat_list)) / (
                          torch.max(sam_img00_feat_list) - torch.min(sam_img00_feat_list) + epsilon)


    he = ed_img00_feat_list #* sam_img00_feat_list
    he = (he - torch.min(he)) / (torch.max(he) - torch.min(he) + epsilon)
    he = he.numpy()

    count_45_to_55 = np.sum(np.logical_and(he >= 0.45, he <= 0.55))

    # print(f"count_45_to_55: {count_45_to_55}")

    results.append(count_45_to_55)



# 将结果从小到大排序并保存到txt文件
sorted_results = sorted(enumerate(results), key=lambda x: x[1])
with open(r"/home/user/zly/daima/UKUQ/result/results_spectral_1_min_max.txt", "w") as f:
    for index, result in sorted_results:
        f.write(f"Image {index}: OA = {result}\n")





# 创建数据集实例
data_dir = "/home/user/zly/daima/train_clip_0"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

results = []

for images_A, images_B, labels in dataloader:
    images_A = torch.stack(images_A)
    images_B = torch.stack(images_B)
    labels = torch.stack(labels)

    #traditional meothod
    epsilon = 1e-8
    dist = (torch.abs(images_A.squeeze(0) - images_B.squeeze(0)) + epsilon) * (
            torch.abs(images_A.squeeze(0) - images_B.squeeze(0)) + epsilon)

    su = torch.sum(dist, dim=0)
    ed_img00_feat_list = torch.sqrt(su + epsilon)
    ed_img00_feat_list = (ed_img00_feat_list - torch.min(ed_img00_feat_list)) / (
            torch.max(ed_img00_feat_list) - torch.min(ed_img00_feat_list) + epsilon)


    d = images_A.squeeze(0) * images_B.squeeze(0)
    d1 = images_A.squeeze(0) * images_A.squeeze(0)
    d2 = images_B.squeeze(0) * images_B.squeeze(0)


    t = torch.sum(d, dim=0)
    t1 = torch.sum(d1, dim=0)
    t2 = torch.sum(d2, dim=0)

    sam_img00_feat_list = (t / torch.clamp((t1 * t2 + epsilon).sqrt(), min=1e-6, max=float('inf')))
    sam_img00_feat_list = torch.max(sam_img00_feat_list)-sam_img00_feat_list
    sam_img00_feat_list = (sam_img00_feat_list - torch.min(sam_img00_feat_list)) / (
                          torch.max(sam_img00_feat_list) - torch.min(sam_img00_feat_list) + epsilon)


    he = ed_img00_feat_list #* sam_img00_feat_list
    he = (he - torch.min(he)) / (torch.max(he) - torch.min(he) + epsilon)
    he = he.numpy()

    count_45_to_55 = np.sum(np.logical_and(he >= 0.45, he <= 0.55))

    # print(f"count_45_to_55: {count_45_to_55}")

    results.append(count_45_to_55)

    # results.append(OA)


sorted_results = sorted(enumerate(results), key=lambda x: x[1], reverse=True)
with open(r"/home/user/zly/daima/UKUQ/result/results_spectral_0_max_min.txt", "w") as f:
    for index, result in sorted_results:
        f.write(f"Image {index}: OA = {result}\n")


end_time = time.time()

execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")