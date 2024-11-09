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


def read_txt(filename, result_dict):
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            index = int(line.split()[1].strip(':'))
            if index in result_dict:
                result_dict[index].append(i + 1)  # Append the line number to the list
            else:
                result_dict[index] = [i + 1]  # Create a new list with the line number



def find_nearest_indices(data):
    # 首先对数据进行排序
    sorted_data = sorted(data)

    # 初始化最小差距和对应的行号对
    min_diff_sum = float('inf')
    nearest_indices = (sorted_data[0], sorted_data[1], sorted_data[2])

    # 遍历所有组合，计算差的绝对值之和
    for i in range(len(sorted_data) - 2):
        for j in range(i + 1, len(sorted_data) - 1):
            for k in range(j + 1, len(sorted_data)):
                diff_sum = abs(sorted_data[i] - sorted_data[j]) + abs(sorted_data[j] - sorted_data[k]) + abs(sorted_data[i] - sorted_data[k])
                if diff_sum < min_diff_sum:
                    min_diff_sum = diff_sum
                    nearest_indices = (sorted_data[i], sorted_data[j], sorted_data[k])

    return nearest_indices  # 返回差值最接近0的行号对


def calculate_nearest_indices(result_dict):
    nearest_indices = {}
    for index, data_list in result_dict.items():
        nearest_indices[index] = find_nearest_indices(data_list)  # Store the nearest indices as a tuple
    return nearest_indices


def main():
    result_dict = {}
    # 从每个txt文件中读取数据
    read_txt(r"D:\zly\daima\sample\gongkai\result\results_classcvaps_0_max_min.txt", result_dict)
    read_txt(r"D:\zly\daima\sample\gongkai\result\results_spectral_0_max_min.txt", result_dict)
    read_txt(r"D:\zly\daima\sample\gongkai\result\result_ssim_index_0_min_max.txt", result_dict)
    read_txt(r"D:\zly\daima\sample\gongkai\result\result_cha_entr_AB_0_max_min.txt", result_dict)
    read_txt(r"D:\zly\daima\sample\gongkai\result\diedai\train_sorted_file_0_gai.txt", result_dict)

    # 计算最接近的行号对
    nearest_indices = calculate_nearest_indices(result_dict)

    # 将结果写入文件
    with open(r"D:\zly\daima\sample\gongkai\result\result_jiejin_0.txt", "w") as f:
        for index, nearest_indices in nearest_indices.items():
            f.write(f"Image {index}: near = {nearest_indices[0]}，{nearest_indices[1]}，{nearest_indices[2]}\n")



    # # Write results to file
    # with open(r"D:\zly\daima\sample\gongkai\result\result_jiejin_0.txt", "w") as f:
    #     for index, (nearest_index1, nearest_index2) in nearest_indices.items():
    #         f.write(f"Image {index}: near = {nearest_index1}，{nearest_index2}\n")


if __name__ == "__main__":
    main()


def main():
    result_dict = {}
    read_txt(r"D:\zly\daima\sample\gongkai\result\results_classcvaps_1_min_max.txt", result_dict)
    read_txt(r"D:\zly\daima\sample\gongkai\result\results_spectral_1_min_max.txt", result_dict)
    read_txt(r"D:\zly\daima\sample\gongkai\result\result_ssim_index_1_min_max.txt", result_dict)
    read_txt(r"D:\zly\daima\sample\gongkai\result\result_cha_entr_AB_1_max_min.txt", result_dict)
    read_txt(r"D:\zly\daima\sample\gongkai\result\diedai\train_sorted_file_1_gai.txt", result_dict)
    nearest_indices = calculate_nearest_indices(result_dict)

    # with open(r"D:\zly\daima\sample\gongkai\result\result_jiejin_1.txt", "w") as f:
    #     for index, (nearest_index1, nearest_index2) in nearest_indices.items():
    #         f.write(f"Image {index}: near = {nearest_index1}，{nearest_index2}\n")

    # 将结果写入文件
    with open(r"D:\zly\daima\sample\gongkai\result\result_jiejin_1.txt", "w") as f:
        for index, nearest_indices in nearest_indices.items():
            f.write(f"Image {index}: near = {nearest_indices[0]}，{nearest_indices[1]}，{nearest_indices[2]}\n")


if __name__ == "__main__":
    main()



end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")






