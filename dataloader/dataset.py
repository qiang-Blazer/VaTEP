import os
import re
import numpy as np
import pandas as pd
import random
from PIL import Image
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader


def frame_transform(p=0.5, img_size=112):
    transform = A.ReplayCompose([
        A.HorizontalFlip(p=p),  
        A.VerticalFlip(p=p),   
        A.Rotate(limit=(-180, 180), p=p),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=p),  
        A.Resize(img_size, img_size),                                                                                                                                                                                                                                                                                              
    ])
    return transform


class FrameAndTable(Dataset):
    def __init__(self, root="datasets/Data", frame_num=48, n=6, img_size=112, seed=0, train=True):
        self.video_root = os.path.join(root, 'videos')
        pos_dirs = [i for i in os.listdir(self.video_root) if i.endswith("1")]
        neg_dirs = [i for i in os.listdir(self.video_root) if i.endswith("0")]
        random.seed(seed)
        random.shuffle(pos_dirs)
        random.shuffle(neg_dirs)
        if train:
            self.video_dirs = pos_dirs[:int(len(pos_dirs)*0.8)] + neg_dirs[:int(len(neg_dirs)*0.8)]
            self.transform = frame_transform(p=0.5, img_size=img_size)
        else:
            self.video_dirs = pos_dirs[int(len(pos_dirs)*0.8):] + neg_dirs[int(len(neg_dirs)*0.8):]
            self.transform = frame_transform(p=0, img_size=img_size)
        random.shuffle(self.video_dirs)

        self.df = pd.read_csv(os.path.join(root, 'infos.csv'), encoding='gbk')
        self.frame_num = frame_num
        self.n = n

    def __len__(self):
        return len(self.video_dirs)

    def extract_frames(self, exp_path, num_frames, n):
        frame_files = sorted(os.listdir(exp_path))
        length = len(frame_files)  
        assert length >= num_frames*n, "Not enough frames in video"
        step_size = length // num_frames
        begin_pos = length % num_frames
        small_step_size = step_size // n

        all_frames = []
        for i in range(n):
            extracted_frame_files = frame_files[(begin_pos+small_step_size*i)::step_size]
            frames = []
            for i in extracted_frame_files:
                frame_path = os.path.join(exp_path, i)
                frame = np.array(Image.open(frame_path).convert("L"))
                frames.append(frame)
            all_frames.append(frames)

        return all_frames

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        result = self.df[self.df['id'] == int(video_dir.split('_')[0])]  
        video_dir = os.path.join(self.video_root, video_dir)
        table = result.iloc[:, 1:40].values.reshape(-1)
        table_cat = torch.from_numpy(table[:3]).int()
        table_con = torch.from_numpy(table[3:]).float()
        FetalHeart_label = result.iloc[:, 40].values
        Multiple_label = result.iloc[:, 41].values
        LiveBirth_label = result.iloc[:, 42].values

        exp_files = [i for i in os.listdir(video_dir) if i.startswith('exp')]
        if len(exp_files) == 1:
            # 有一个文件，重复两次
            exp_path = os.path.join(video_dir, exp_files[0])
            all_frames = 2*self.extract_frames(exp_path, num_frames=self.frame_num, n=self.n)
        elif len(exp_files) == 2:
            # 有两个文件
            all_frames = []
            for file in exp_files:
                exp_path = os.path.join(video_dir, file)
                temp_frames = self.extract_frames(exp_path, num_frames=self.frame_num, n=self.n)
                all_frames += temp_frames
        else:
            raise Exception("Unexpected number of exp files")

        all_frames_input = []
        for frames in all_frames:
            augmented_frame = self.transform(image=frames[0])
            replay_params = augmented_frame['replay']
            transformed_frames = [augmented_frame['image']]
            for i in range(1, len(frames)):
                augmented_frame = A.ReplayCompose.replay(replay_params, image=frames[i])
                transformed_frames.append(augmented_frame['image'])
            frames = np.stack(transformed_frames)  #[T, H, W]
            frames = torch.from_numpy(frames).float() / 255.0    #[T, H, W]
            all_frames_input.append(frames)
        all_frames_input = torch.stack(all_frames_input)  #[n, T, H, W]

        return all_frames_input, table_cat, table_con, FetalHeart_label, Multiple_label, LiveBirth_label


if __name__ == '__main__':
    dataset = FrameAndTable(frame_num=32, n=6, img_size=64, seed=0, train=True)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    print(len(dataset))
    for i in data_loader:
        print(i[0].shape, i[1].shape, i[2].shape, i[3], i[4], i[5])