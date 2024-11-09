import os
import cv2
import re
import time

start_time = time.time()

train_file_path = r'D:/zly/daima/sample/gongkai/result/diedai/train_file.txt'  # 替换为实际的train_file.txt路径
label_folder = 'D:\zly\data\WHU_clip/all/train\label/'  # 替换为实际的label文件夹路径
output_folder_1 = r'D:\zly\daima\sample\gongkai\result\diedai/'  # 替换为第一个输出文件夹路径
output_folder_2 = r'D:\zly\daima\sample\gongkai\result\diedai/'  # 替换为第二个输出文件夹路径

# 读取train_file.txt文件内容并处理
def process_train_file(train_file_path):
    with open(train_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            filename_A = line.strip().split(': ')[0].split('Image ')[1]
            pixel_count = int(re.search(r'pixel_count = (\d+)', line).group(1))
            img_path = os.path.join(label_folder, filename_A)
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if (img == 0).all():
                    with open(os.path.join(output_folder_1, 'train_file_0.txt'), 'a') as out_file_1:
                        out_file_1.write(f"{filename_A}: pixel_count = {pixel_count}\n")
                else:
                    with open(os.path.join(output_folder_2, 'train_file_1.txt'), 'a') as out_file_2:
                        out_file_2.write(f"{filename_A}: pixel_count = {pixel_count}\n")

process_train_file(train_file_path)


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_file_0.txt', 'r') as file:
    lines = file.readlines()
# 遍历每一行并赋值pixel_count为对应行号
for i, line in enumerate(lines):
    filename_A, pixel_count = line.split(': pixel_count = ')
    pixel_count = str(i + 1)
    lines[i] = f"{filename_A}: pixel_count = {pixel_count}\n"

# 将修改后的结果写入新文件
with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_file_0.txt', 'w') as out_file:
    out_file.write(''.join(lines))



lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_file_1.txt', 'r') as file:
    lines = file.readlines()


# 遍历每一行并赋值pixel_count为对应行号
for i, line in enumerate(lines):
    filename_A, pixel_count = line.split(': pixel_count = ')
    pixel_count = str(i + 1)
    lines[i] = f"{filename_A}: pixel_count = {pixel_count}\n"

# 将修改后的结果写入新文件
with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_file_1.txt', 'w') as out_file:
    out_file.write(''.join(lines))



end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")
