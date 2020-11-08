import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import *


class SteelData(Dataset):
    def __init__(self, root, mode='train', csv=None):
        super(SteelData, self).__init__()
        img_ids = list(set(csv['ImageId'].tolist()))
        img_ids.sort()
        pos = int(len(img_ids) * 0.8)
        self.root = root
        self.csv = csv
        if mode == 'train':
            self.img_ids = img_ids[:pos]
            self.transform = Compose([
                ToPILImage(),
                ColorJitter(0.4, 0.3, 0.3),
                ToTensor(),
                RandomErasing(p=0.5, scale=(0.03, 0.1), ratio=(0.05, 0.4)),
            ])
        else:
            self.img_ids = img_ids[pos:]
            self.transform = ToTensor()

    def decode(self, rows):
        '''
        ImageId,ClassId,EncodedPixels
        '''
        mask = np.zeros((256*1600), np.uint8)
        for j in range(len(rows)):
            row = rows.iloc[j]
            class_id = row['ClassId']
            encoded_pixels = np.array([int(ele)
                                       for ele in row['EncodedPixels'].split(' ')])
            starts, lengths = encoded_pixels[::2], encoded_pixels[1::2]
            starts -= 1  # 因为起始值是1，所以先要把坐标减一下
            for index, start in enumerate(starts):
                mask[int(start):int(start + lengths[index])] = class_id
        mask = mask.reshape((1600, 256)).T
        my_mask = np.zeros((256, 1600, 5))
        my_mask[:, :, 0] = 1
        my_mask[:, :, 0][mask > 0] = 0
        my_mask[:, :, 1][mask == 1] = 1
        my_mask[:, :, 2][mask == 2] = 1
        my_mask[:, :, 3][mask == 3] = 1
        my_mask[:, :, 4][mask == 4] = 1

        return my_mask

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        rows = self.csv[self.csv['ImageId'] == img_id]
        mask = self.decode(rows)
        img = cv2.imread(os.path.join(self.root, 'train_images', img_id))
        img = self.transform(img)
        mask = torch.from_numpy(mask).permute(2,0,1)
        return img, mask
