import os

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvison.transforms import *
from util.label_util import LabelLoader


class SteelData(Dataset):
    def __init__(self, root, mode='train'):
        super(SteelData, self).__init__()
        train_csv = pd.read_csv('data/steel/train.csv')
        pos = int(len(train_csv) * 0.8)
        self.root = root
        if mode == 'train':
            self.csv = train_csv[:pos]
            self.label_loader = LabelLoader(self.csv)
            self.label_loader.run()
            self.transform = Compose([
                ToPILImage(),
                ColorJitter(0.4, 0.3, 0.3),
                ToTensor(),
                RandomErasing(p=0.5, scale=(0.03, 0.1), ratio=(0.05, 0.4)),
            ])
        else:
            self.csv = train_csv[pos:]
            self.label_loader = LabelLoader(self.csv)
            self.label_loader.run()
            self.transform = ToTensor()

    def __len__(self):
        return len(self.label_loader.mask_map)

    def __getitem__(self, index):
        row = self.csv[index]
        img_id = row['ImageId']
        mask = self.label_loader.mask_map[img_id]
        img = cv2.imread(os.path.join(self.root, 'train_images', img_id))
        img = self.transform(img)
        return img, ToTensor()(mask)
