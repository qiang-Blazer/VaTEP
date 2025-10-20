import os
import re
import numpy as np
import pandas as pd
import cv2
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader


def frame_transform(p=0.5, img_size=112):
    transform = A.ReplayCompose([
        A.HorizontalFlip(p=p),    # 水平翻转，概率为p
        A.VerticalFlip(p=p),      # 垂直翻转，概率为p
        A.Rotate(limit=(-180, 180), p=p),  # 旋转，旋转角度范围为-180到180，概率为p
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=p),    # 色彩抖动，亮度、对比度、饱和度、色调的变化范围，概率为p
        A.Resize(img_size, img_size),    # 调整图像大小为112x112
    ])
    return transform

# 定义一个函数，用于提取文件名中“RUN”之后的数字
def extract_run_number(filename):
    # 使用正则表达式在文件名中查找RUN后面的数字
    match = re.search(r'img_(\d+)', filename, re.IGNORECASE)
    if match:
        # 如果找到匹配的数字，返回该数字（作为整数）
        return int(match.group(1))
    else:
        # 如果没有找到匹配的数字，返回一个很大的数，以便这些文件排在最后
        return float('inf')


phase_names = {'tPB2':1, 'tPNa':2, 'tPNf':3, 't2':4,
               't3':5,   't4':6,   't5':7,   't6':8, 
               't7':9,   't8':10,  'others':11}


class FrameAndPhase(Dataset):
    def __init__(self, root="datasets/Data_pretrain", frame_num=48, img_size=112, seed=0, train=True):
        self.root = root
        self.persons = os.listdir(self.root)
        if train:
            self.persons = self.persons[:int(len(self.persons)*0.8)]
            self.transform = frame_transform(p=0.5, img_size=img_size)
        else:
            self.persons = self.persons[int(len(self.persons)*0.8):]
            self.transform = frame_transform(p=0.0, img_size=img_size)
        random.seed(seed)
        random.shuffle(self.persons)

        self.frame_num = frame_num

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        video_dir = os.path.join(self.root, self.persons[idx], 'frames')
        all_frame_files = sorted(os.listdir(video_dir), key=extract_run_number)
        all_frames_num = len(all_frame_files)
        random_numbers = random.sample(range(0, all_frames_num), self.frame_num)
        random_numbers.sort()
        frames = []
        for i in random_numbers:
            frame_file = os.path.join(video_dir, all_frame_files[i])
            frames.append(np.array(Image.open(frame_file).convert("L")))
        
        annot_file = os.path.join(self.root, self.persons[idx], 'phases.csv')
        annot_df = pd.read_csv(annot_file, header=None)
        phase2tick = {row[0]: row[1:].tolist() for index, row in annot_df.iterrows()}
        labels = np.array([12]*all_frames_num)
        for phase, ticks in phase2tick.items():
            if phase in phase_names:
                labels[(ticks[0]-1):ticks[1]] = phase_names[phase]
        i = 0
        while i < self.frame_num and labels[i] == 12:
            labels[i] = 0
            i += 1
        labels = labels[random_numbers]

        augmented_frame = self.transform(image=frames[0])
        replay_params = augmented_frame['replay']
        transformed_frames = [augmented_frame['image']]
        for i in range(1, len(frames)):
            augmented_frame = A.ReplayCompose.replay(replay_params, image=frames[i])
            transformed_frames.append(augmented_frame['image'])
        frames = np.stack(transformed_frames)  #[T, H, W]
        frames = torch.from_numpy(frames).float() / 255.0    #[T, H, W]

        return frames, torch.tensor(labels, dtype=torch.long)


if __name__ == '__main__':
    dataset = FrameAndPhase(frame_num=32, img_size=56, seed=0, train=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(len(dataset))
    for i in data_loader:
        print(i[0].shape, i[1])

        