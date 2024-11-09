
from PIL import Image

import math

import collections

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

from utils.datasets import ATDataset

from torchvision.transforms import Resize, CenterCrop, Normalize
from utils.metrics import Metrics


from models.unet.UNet_CD import Unet

import datetime
import random
import os
import tqdm

import argparse

from torch_poly_lr_decay import PolynomialLRDecay
import numpy as np

import time

start_time = time.time()

device = 'cuda'
path = r'D:\zly\daima\sample\gongkai\result\diedai/xuanqu_diedai/t0.90_0.90/'

seed = 66  # 可以选择任意整数作为种子
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_dataset_loaders(workers, batch_size=4):
    target_size = 256

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    trainval_transform = transforms.Compose(
        [

            transforms.ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    target_transform = transforms.Compose(
        [

            transforms.ToTensor(),

        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            Normalize(mean=mean, std=std)
        ]
    )

    train_dataset = ATDataset(
        os.path.join(path, "train", "A"), os.path.join(path, "train", "B"), os.path.join(path, "train", "label"),
        trainval_transform, test_transform, target_transform
    )



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)


    return train_loader

pixel_count = []
def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.train()
    for images1, images2, masks in tqdm.tqdm(loader):
        images1 = images1.to(device)
        images2 = images2.to(device)
        masks = masks.to(device)

        assert images1.size()[2:] == images2.size()[2:] == masks.size()[
                                                           2:], "resolutions for images and masks are in sync"

        num_samples += int(images1.size(0))

        optimizer.zero_grad()
        outputs = net(images1, images2)


        assert outputs.size()[2:] == masks.size()[2:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        outputs = outputs[:, 0, :, :]#.unsqueeze(1)

        output_np = outputs.cpu().detach().numpy()
        for output in output_np:
            count = np.sum((output > 0.45) & (output < 0.55))
            pixel_count.append(count)

        sorted_results = sorted(enumerate(pixel_count), key=lambda x: x[1], reverse=False)
        with open(r"D:\zly\daima\sample\gongkai\result\diedai/pixel_count_train_0.90_paixu_min_max.txt", "w") as f:
            for index, result in sorted_results:
                f.write(f"Image {index}: pixel_count = {result}\n")




        masks = masks[:, 0, :, :]#.unsqueeze(1)


        loss = criterion(outputs, masks.float())

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    assert num_samples > 0, "dataset contains training images and labels"

    return {
        "loss": running_loss / num_samples,
        "precision": metrics.get_precision(),
        "recall": metrics.get_recall(),
        "f_score": metrics.get_f_score(),
        "oa": metrics.get_oa()
    }

pixel_count1 = []



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                        help='Batch Size')
    parser.add_argument('--swa_start', nargs='?', type=int, default=1)
    parser.add_argument('--lr', nargs='?', type=float, default=5e-3,
                        help='Learning Rate')
    parser.add_argument('--model', nargs='?', type=str, default='Unet')
    parser.add_argument('--swa', nargs='?', type=bool, default=True)

    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--pretrain', default=True, type=bool)

    parser.add_argument('--r', dest='resume', default=False, type=bool)

    arg = parser.parse_args()

    num_classes = 2
    model_name = arg.model
    print(model_name)
    learning_rate = arg.lr
    num_epochs = arg.n_epoch
    batch_size = arg.batch_size

    history = collections.defaultdict(list)
    model_dict = {

        'Unet': Unet(input_nbr=6, label_nbr=2).train().to(device),
    }

    net = model_dict[model_name]
    print(net)
    if torch.cuda.device_count() > 1:
        print("using multi gpu")
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    else:
        print('using one gpu')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_loader = get_dataset_loaders(5, batch_size)
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate)

    today = str(datetime.date.today())


    scheduler = PolynomialLRDecay(opt, max_decay_steps=100, end_learning_rate=0.0001, power=2.0)

    for epoch in range(num_epochs):

        scheduler.step()

        train_hist = train(train_loader, num_classes, device, net, opt, criterion)

        f1score = train_hist["f_score"]
        for k, v in train_hist.items():
            history["train " + k].append(v)


























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
data_dir = r"D:\zly\daima\sample\gongkai\result\diedai\xuanqu_diedai/t0.90_0.90/train"
transform = transforms.ToTensor()
dataset = ChangeDetectionDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn)

# 读取 final_result.txt 文件中的图像索引
index_list = []
with open(r'D:\zly\daima\sample\gongkai\result\diedai\pixel_count_train_0.90_paixu_min_max.txt', 'r') as file:
    for line in file:
        if line.startswith('Image'):
            index = int(line.split()[1].strip(':'))
            index_list.append(index)

# num_samples = 7120 #len(index_list)
num_selected = math.ceil(4003 * 0.80) + math.ceil(1331 * 0.80)
selected_indices = index_list[:num_selected]

# 保存对应索引的图像
output_folder = r"D:\zly\daima\sample\gongkai\result\diedai\xuanqu_diedai\t0.80_0.80\train"  # 修复文件夹路径错误
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


end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"代码执行时间: {execution_time:.6f} 秒")
