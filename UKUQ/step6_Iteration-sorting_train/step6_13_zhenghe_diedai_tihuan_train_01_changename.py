import os

import time

start_time = time.time()

sorted_file_path = r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_file_0_gai.txt'
image_folder = 'D:/zly/data/WHU_clip/train_clip_0/label/'

lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_file_0.txt', 'r') as file:
    lines = file.readlines()

# 遍历每一行并赋值pixel_count为对应行号
for i, line in enumerate(lines):
    split_line = line.split(': pixel_count = ')
    if len(split_line) == 2:
        filename_A, pixel_count = split_line
        filename_A = filename_A.strip()  # 去除文件名中可能存在的空格
        matching_images = [f for f in os.listdir(image_folder)] #if f.startswith(filename_A)]
        matching_images.sort()  # 对文件名进行排序
        image_order = matching_images.index(filename_A ) #if (filename_A) in matching_images else 0
        lines[i] = f"Image {image_order}: sum of line numbers = {pixel_count}"


# 将修改后的结果写入新文件
with open(sorted_file_path, 'w') as out_file:
    out_file.write(''.join(lines))

















import os

sorted_file_path = r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_file_1_gai.txt'
image_folder = 'D:/zly/data/WHU_clip/train_clip_1/label/'

lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_file_1.txt', 'r') as file:
    lines = file.readlines()

# 遍历每一行并赋值pixel_count为对应行号
for i, line in enumerate(lines):
    split_line = line.split(': pixel_count = ')
    if len(split_line) == 2:
        filename_A, pixel_count = split_line
        filename_A = filename_A.strip()  # 去除文件名中可能存在的空格
        matching_images = [f for f in os.listdir(image_folder)] #if f.startswith(filename_A)]
        matching_images.sort()  # 对文件名进行排序
        image_order = matching_images.index(filename_A ) #if (filename_A) in matching_images else 0
        lines[i] = f"Image {image_order}: sum of line numbers = {pixel_count}"


# 将修改后的结果写入新文件
with open(sorted_file_path, 'w') as out_file:
    out_file.write(''.join(lines))



end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")
