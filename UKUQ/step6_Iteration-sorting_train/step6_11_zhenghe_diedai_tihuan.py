import time

start_time = time.time()

# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.10.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.20.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.20.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.20.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_0.20.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+713}\n")



# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.20.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.30.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.30.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.30.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_0.30.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+1425}\n")






# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.30.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.40.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.40.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.40.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_0.40.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+2137}\n")






# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.40.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.50.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.50.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.50.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_0.50.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+2849}\n")






# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.50.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.60.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.60.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.60.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_0.60.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+3561}\n")






# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.60.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.70.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.70.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.70.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_0.70.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+4273}\n")






# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.70.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.80.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.80.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.80.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_0.80.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+4985}\n")






# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.80.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.90.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.90.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_0.90.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_0.90.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+5697}\n")






# 读取1.txt的内容，提取filename_A
filename_A_list = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_0.90.txt', 'r') as file:
    content_1 = file.readlines()
    for line in content_1:
        if "Image" in line:
            filename_A = line.split(":")[0].strip().replace("Image ", "")
            filename_A_list.append(filename_A)

# 读取2.txt的内容并处理
lines_to_export = []
lines_with_pixel_count = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/diedai_train_all.txt', 'r') as file:
    content_2 = file.readlines()
    for line in content_2:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        line_number = parts[0]#.strip().replace("Image ", "").split("_")[-1]
        if any(filename_A in line for filename_A in filename_A_list):
            lines_to_export.append(line)
        else:
            try:
                pixel_count = int(parts[1].split("=")[-1].strip())
                lines_with_pixel_count.append(((line_number), pixel_count))
            except ValueError:
                continue


# 对2.txt中剩余filename_A按行号从小到大对pixel_count的赋值，并导出到4.txt的python代码
sorted_lines = sorted(lines_with_pixel_count, key=lambda x: x[0])
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_all.txt', 'w') as file:
    for line_number, pixel_count in sorted_lines:
        filename_A = f"{line_number}"
        file.write(f"{filename_A}: pixel_count = {pixel_count}\n")


lines = []
with open(r'D:/zly/daima/sample/gongkai/result/diedai/shiyu_all.txt', 'r') as file:
    lines = file.readlines()

# 提取每行中的 pixel_count，并将行号与 pixel_count 组成元组存储在列表中
pixel_counts = []
for line in lines:
    parts = line.split('=')
    filename_A = parts[0].split(':')[0].strip()
    pixel_count = int(parts[-1].strip())
    pixel_counts.append((filename_A, pixel_count))

# 按照 pixel_count 从小到大排序
sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1])

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_sorted_all.txt', 'w') as file:
    for index, (filename_A, pixel_count) in enumerate(sorted_pixel_counts, start=1):
        file.write(f"{filename_A}: pixel_count = {index+6409}\n")



input_files = [r'D:\zly\daima\sample\gongkai\result\diedai/diedai_train_0.10.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_0.20.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_0.30.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_0.40.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_0.50.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_0.60.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_0.70.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_0.80.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_0.90.txt',
               r'D:\zly\daima\sample\gongkai\result\diedai/train_sorted_all.txt']  # 替换为你要合并的文件列表

with open(r'D:/zly/daima/sample/gongkai/result/diedai/train_file.txt', 'w') as outfile:
    for fname in input_files:
        with open(fname) as infile:
            outfile.write(infile.read())



end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")





