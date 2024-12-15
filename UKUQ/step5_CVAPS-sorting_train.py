import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
start_time = time.time()

class ChangeDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files_A = sorted(os.listdir(os.path.join(data_dir, 'images_png')))
        # self.image_files_B = sorted(os.listdir(os.path.join(data_dir, 'B')))
        self.image_files_label = sorted(os.listdir(os.path.join(data_dir, 'masks_png')))
        self.transform = transform

    def __getitem__(self, index):
        filename_A = os.path.join(self.data_dir, 'images_png', self.image_files_A[index])
        # filename_B = os.path.join(self.data_dir, 'B', self.image_files_B[index])
        filename_path = os.path.join(self.data_dir, 'masks_png', self.image_files_label[index])

        image_A = Image.open(filename_A)  # .convert("RGB")
        # image_B = Image.open(filename_B).convert("RGB")
        label = Image.open(filename_path)  # .convert("RGB")

        if self.transform is not None:
            image_A = self.transform(image_A)
            # image_B = self.transform(image_B)
            label = self.transform(label)

        return image_A, label

    def __len__(self):
        return len(self.image_files_A)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

results_2 = []
result_1 = []
result_3 = []
result_4 = []

Image.MAX_IMAGE_PIXELS = None

# # 创建数据集实例
# data_dir = r"D:\zly\2021LoveDA\Urban"
# transform = transforms.ToTensor()
# dataset = ChangeDetectionDataset(data_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
#
# total_mean1 = torch.zeros(3)
# total_mean2 = torch.zeros(3)
# total_mean3 = torch.zeros(3)
# total_mean4 = torch.zeros(3)
# total_mean5 = torch.zeros(3)
# total_mean6 = torch.zeros(3)
# total_mean7 = torch.zeros(3)
#
# count1 = 0
# count2 = 0
# count3 = 0
# count4 = 0
# count5 = 0
# count6 = 0
# count7 = 0
#
# target_label1 = 1
# target_label2 = 2
# target_label3 = 3
# target_label4 = 4
# target_label5 = 5
# target_label6 = 6
# target_label7 = 7
#
# for images_A, labels in dataloader:
#     images_A = torch.stack(images_A)
#     labels = torch.stack(labels)
#     images_A = images_A
#     labels = labels * 255
#
#
#     labels = labels.squeeze(0)
#
#     # 根据label值筛选图像
#     mask1 = torch.where(labels == target_label1, torch.tensor(1), torch.tensor(0))
#     filtered_images_A = images_A * mask1.unsqueeze(1)
#     # 计算筛选后图像的平均值
#     mean1 = torch.mean(filtered_images_A, dim=(0, 2, 3))
#     total_mean1 += mean1
#     count1 += 1
#
#     # 根据label值筛选图像
#     mask2 = torch.where(labels == target_label2, torch.tensor(1), torch.tensor(0))
#     filtered_images_A = images_A * mask2.unsqueeze(1)
#     # 计算筛选后图像的平均值
#     mean2 = torch.mean(filtered_images_A, dim=(0, 2, 3))
#     total_mean2 += mean2
#     count2 += 1
#
#     # 根据label值筛选图像
#     mask3 = torch.where(labels == target_label3, torch.tensor(1), torch.tensor(0))
#     filtered_images_A = images_A * mask3.unsqueeze(1)
#     # 计算筛选后图像的平均值
#     mean3 = torch.mean(filtered_images_A, dim=(0, 2, 3))
#     total_mean3 += mean3
#     count3 += 1
#
#     # 根据label值筛选图像
#     mask4 = torch.where(labels == target_label4, torch.tensor(1), torch.tensor(0))
#     filtered_images_A = images_A * mask4.unsqueeze(1)
#     # 计算筛选后图像的平均值
#     mean4 = torch.mean(filtered_images_A, dim=(0, 2, 3))
#     total_mean4 += mean4
#     count4 += 1
#
#     # 根据label值筛选图像
#     mask5 = torch.where(labels == target_label5, torch.tensor(1), torch.tensor(0))
#     filtered_images_A = images_A * mask5.unsqueeze(1)
#     # 计算筛选后图像的平均值
#     mean5 = torch.mean(filtered_images_A, dim=(0, 2, 3))
#     total_mean5 += mean5
#     count5 += 1
#
#     # 根据label值筛选图像
#     mask6 = torch.where(labels == target_label6, torch.tensor(1), torch.tensor(0))
#     filtered_images_A = images_A * mask6.unsqueeze(1)
#     # 计算筛选后图像的平均值
#     mean6 = torch.mean(filtered_images_A, dim=(0, 2, 3))
#     total_mean6 += mean6
#     count6 += 1
#
#     # 根据label值筛选图像
#     mask7 = torch.where(labels == target_label7, torch.tensor(1), torch.tensor(0))
#     filtered_images_A = images_A * mask7.unsqueeze(1)
#     # 计算筛选后图像的平均值
#     mean7 = torch.mean(filtered_images_A, dim=(0, 2, 3))
#     total_mean7 += mean7
#     count7 += 1
#
# # 计算平均值的平均值
# average_mean1 = total_mean1 / count1
# average_mean2 = total_mean2 / count2
# average_mean3 = total_mean3 / count3
# average_mean4 = total_mean4 / count4
# average_mean5 = total_mean5 / count5
# average_mean6 = total_mean6 / count6
# average_mean7 = total_mean7 / count7
#
# print("Average Mean for label {}: {}".format(target_label1, average_mean1))
# print("Average Mean for label {}: {}".format(target_label2, average_mean2))
# print("Average Mean for label {}: {}".format(target_label3, average_mean3))
# print("Average Mean for label {}: {}".format(target_label4, average_mean4))
# print("Average Mean for label {}: {}".format(target_label5, average_mean5))
# print("Average Mean for label {}: {}".format(target_label6, average_mean6))
# print("Average Mean for label {}: {}".format(target_label7, average_mean7))



average_mean1 = torch.tensor([0.1394, 0.1470, 0.1438])
average_mean2 = torch.tensor([0.0750, 0.0779, 0.0782])
average_mean3 = torch.tensor([0.0291, 0.0304, 0.0301])
average_mean4 = torch.tensor([0.0083, 0.0096, 0.0095])
average_mean5 = torch.tensor([0.0200, 0.0208, 0.0199])
average_mean6 = torch.tensor([0.0156, 0.0191, 0.0180])
average_mean7 = torch.tensor([0.0052, 0.0056, 0.0052])

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




Image.MAX_IMAGE_PIXELS = None

# 创建数据集实例
data_dir = r"/home/user/zly/daima/train_clip_1"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

for images_A, images_B, labels in dataloader:
    images_A = torch.stack(images_A)
    images_B = torch.stack(images_B)
    labels = torch.stack(labels)
    images_A = images_A.squeeze(0).numpy()
    images_B = images_B.squeeze(0).numpy()


    acankao1 = np.zeros((3, 256, 256))
    acankao1[0, :, :] = average_mean1[0]
    acankao1[1, :, :] = average_mean1[1]
    acankao1[2, :, :] = average_mean1[2]

    acankao2 = np.zeros((3, 256, 256))
    acankao2[0, :, :] = average_mean2[0]
    acankao2[1, :, :] = average_mean2[1]
    acankao2[2, :, :] = average_mean2[2]

    acankao3 = np.zeros((3, 256, 256))
    acankao3[0, :, :] = average_mean3[0]
    acankao3[1, :, :] = average_mean3[1]
    acankao3[2, :, :] = average_mean3[2]

    acankao4 = np.zeros((3, 256, 256))
    acankao4[0, :, :] = average_mean4[0]
    acankao4[1, :, :] = average_mean4[1]
    acankao4[2, :, :] = average_mean4[2]

    acankao5 = np.zeros((3, 256, 256))
    acankao5[0, :, :] = average_mean5[0]
    acankao5[1, :, :] = average_mean5[1]
    acankao5[2, :, :] = average_mean5[2]

    acankao6 = np.zeros((3, 256, 256))
    acankao6[0, :, :] = average_mean6[0]
    acankao6[1, :, :] = average_mean6[1]
    acankao6[2, :, :] = average_mean6[2]

    acankao7 = np.zeros((3, 256, 256))
    acankao7[0, :, :] = average_mean7[0]
    acankao7[1, :, :] = average_mean7[1]
    acankao7[2, :, :] = average_mean7[2]

    change_cha1 = (images_A - acankao1) * (images_A - acankao1)
    change_cha1 = torch.from_numpy(change_cha1)
    cva1_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao2) * (images_A - acankao2)
    change_cha1 = torch.from_numpy(change_cha1)
    cva2_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao3) * (images_A - acankao3)
    change_cha1 = torch.from_numpy(change_cha1)
    cva3_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao4) * (images_A - acankao4)
    change_cha1 = torch.from_numpy(change_cha1)
    cva4_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao5) * (images_A - acankao5)
    change_cha1 = torch.from_numpy(change_cha1)
    cva5_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao6) * (images_A - acankao6)
    change_cha1 = torch.from_numpy(change_cha1)
    cva6_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao7) * (images_A - acankao7)
    change_cha1 = torch.from_numpy(change_cha1)
    cva7_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao1) * (images_B - acankao1)
    change_cha1 = torch.from_numpy(change_cha1)
    cva1_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao2) * (images_B - acankao2)
    change_cha1 = torch.from_numpy(change_cha1)
    cva2_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao3) * (images_B - acankao3)
    change_cha1 = torch.from_numpy(change_cha1)
    cva3_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao4) * (images_B - acankao4)
    change_cha1 = torch.from_numpy(change_cha1)
    cva4_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao5) * (images_B - acankao5)
    change_cha1 = torch.from_numpy(change_cha1)
    cva5_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao6) * (images_B - acankao6)
    change_cha1 = torch.from_numpy(change_cha1)
    cva6_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao7) * (images_B - acankao7)
    change_cha1 = torch.from_numpy(change_cha1)
    cva7_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    imagA = [cva1_A, cva2_A, cva3_A, cva4_A, cva5_A, cva6_A, cva7_A]
    imagA = np.stack(imagA, axis=0)
    imagB = [cva1_B, cva2_B, cva3_B, cva4_B, cva5_B, cva6_B, cva7_B]
    imagB = np.stack(imagB, axis=0)
    change_cha1 = (imagA - imagB) * (imagA - imagB)
    change_cha1 = torch.from_numpy(change_cha1)
    cvacva = torch.sqrt( change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :] + change_cha1[3, :, :] + change_cha1[4, :,:] + change_cha1[5, :, :] + change_cha1[ 6, :, :])

    epsilon = 1e-8
    he = cvacva
    he = (he - torch.min(he)) / (torch.max(he) - torch.min(he) + epsilon)
    he = he.numpy()

    count_45_to_55 = np.sum(np.logical_and(he >= 0.45, he <= 0.55))
    # print(f"count_45_to_55: {count_45_to_55}")


    results_2.append(count_45_to_55)

# 将结果从小到大排序并保存到txt文件 OA 越大越准确 越小越困难
sorted_results = sorted(enumerate(results_2), key=lambda x: x[1])
with open(r"/home/user/zly/daima/UKUQ/result/results_classcvaps_1_min_max.txt", "w") as f:
    for index, result in sorted_results:
        f.write(f"Image {index}: OA = {result}\n")






results_2 = []
result_1 = []
result_3 = []
result_4 = []


average_mean1 = torch.tensor([0.1394, 0.1470, 0.1438])
average_mean2 = torch.tensor([0.0750, 0.0779, 0.0782])
average_mean3 = torch.tensor([0.0291, 0.0304, 0.0301])
average_mean4 = torch.tensor([0.0083, 0.0096, 0.0095])
average_mean5 = torch.tensor([0.0200, 0.0208, 0.0199])
average_mean6 = torch.tensor([0.0156, 0.0191, 0.0180])
average_mean7 = torch.tensor([0.0052, 0.0056, 0.0052])



Image.MAX_IMAGE_PIXELS = None

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


    acankao1 = np.zeros((3, 256, 256))
    acankao1[0, :, :] = average_mean1[0]
    acankao1[1, :, :] = average_mean1[1]
    acankao1[2, :, :] = average_mean1[2]

    acankao2 = np.zeros((3, 256, 256))
    acankao2[0, :, :] = average_mean2[0]
    acankao2[1, :, :] = average_mean2[1]
    acankao2[2, :, :] = average_mean2[2]

    acankao3 = np.zeros((3, 256, 256))
    acankao3[0, :, :] = average_mean3[0]
    acankao3[1, :, :] = average_mean3[1]
    acankao3[2, :, :] = average_mean3[2]

    acankao4 = np.zeros((3, 256, 256))
    acankao4[0, :, :] = average_mean4[0]
    acankao4[1, :, :] = average_mean4[1]
    acankao4[2, :, :] = average_mean4[2]

    acankao5 = np.zeros((3, 256, 256))
    acankao5[0, :, :] = average_mean5[0]
    acankao5[1, :, :] = average_mean5[1]
    acankao5[2, :, :] = average_mean5[2]

    acankao6 = np.zeros((3, 256, 256))
    acankao6[0, :, :] = average_mean6[0]
    acankao6[1, :, :] = average_mean6[1]
    acankao6[2, :, :] = average_mean6[2]

    acankao7 = np.zeros((3, 256, 256))
    acankao7[0, :, :] = average_mean7[0]
    acankao7[1, :, :] = average_mean7[1]
    acankao7[2, :, :] = average_mean7[2]

    change_cha1 = (images_A - acankao1) * (images_A - acankao1)
    change_cha1 = torch.from_numpy(change_cha1)
    cva1_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao2) * (images_A - acankao2)
    change_cha1 = torch.from_numpy(change_cha1)
    cva2_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao3) * (images_A - acankao3)
    change_cha1 = torch.from_numpy(change_cha1)
    cva3_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao4) * (images_A - acankao4)
    change_cha1 = torch.from_numpy(change_cha1)
    cva4_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao5) * (images_A - acankao5)
    change_cha1 = torch.from_numpy(change_cha1)
    cva5_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao6) * (images_A - acankao6)
    change_cha1 = torch.from_numpy(change_cha1)
    cva6_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_A - acankao7) * (images_A - acankao7)
    change_cha1 = torch.from_numpy(change_cha1)
    cva7_A = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao1) * (images_B - acankao1)
    change_cha1 = torch.from_numpy(change_cha1)
    cva1_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao2) * (images_B - acankao2)
    change_cha1 = torch.from_numpy(change_cha1)
    cva2_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao3) * (images_B - acankao3)
    change_cha1 = torch.from_numpy(change_cha1)
    cva3_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao4) * (images_B - acankao4)
    change_cha1 = torch.from_numpy(change_cha1)
    cva4_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao5) * (images_B - acankao5)
    change_cha1 = torch.from_numpy(change_cha1)
    cva5_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao6) * (images_B - acankao6)
    change_cha1 = torch.from_numpy(change_cha1)
    cva6_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    change_cha1 = (images_B - acankao7) * (images_B - acankao7)
    change_cha1 = torch.from_numpy(change_cha1)
    cva7_B = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :])

    imagA = [cva1_A, cva2_A, cva3_A, cva4_A, cva5_A, cva6_A, cva7_A]
    imagA = np.stack(imagA, axis=0)
    imagB = [cva1_B, cva2_B, cva3_B, cva4_B, cva5_B, cva6_B, cva7_B]
    imagB = np.stack(imagB, axis=0)
    change_cha1 = (imagA - imagB) * (imagA - imagB)
    change_cha1 = torch.from_numpy(change_cha1)
    cvacva = torch.sqrt(change_cha1[0, :, :] + change_cha1[1, :, :] + change_cha1[2, :, :] + change_cha1[3, :, :] + change_cha1[4, :,:] + change_cha1[5, :, :] + change_cha1[ 6, :, :])

    epsilon = 1e-8
    he = cvacva
    he = (he - torch.min(he)) / (torch.max(he) - torch.min(he) + epsilon)
    he = he.numpy()

    count_45_to_55 = np.sum(np.logical_and(he >= 0.45, he <= 0.55))
    # print(f"count_45_to_55: {count_45_to_55}")


    results_2.append(count_45_to_55)


sorted_results = sorted(enumerate(results_2), key=lambda x: x[1], reverse=True)
with open(r"/home/user/zly/daima/UKUQ/result/results_classcvaps_0_max_min.txt", "w") as f:
    for index, result in sorted_results:
        f.write(f"Image {index}: OA = {result}\n")

end_time = time.time()

execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")