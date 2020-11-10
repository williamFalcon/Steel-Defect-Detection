import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import *

from util.augment import Augmentor





class SteelData(Dataset):
    def __init__(self, root, mode='train', csv=None, width=512, height=256, num_classes=4):
        super(SteelData, self).__init__()
        img_ids = list(set(csv['ImageId'].tolist()))
        img_ids.sort()
        pos = int(len(img_ids) * 0.8)
        self.num_classes = num_classes
        self.root = root
        self.csv = csv
        self.normalize = Compose([
            ToPILImage(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])])
        augmentor = Augmentor(width=width, height=height)
        if mode == 'train':
            self.img_ids = img_ids[: pos]
            self.transform = augmentor.aug_train
        else:
            self.img_ids = img_ids[pos:]
            self.transform = augmentor.aug_val

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
            starts, lengths = encoded_pixels[:: 2], encoded_pixels[1:: 2]
            starts -= 1  # 因为起始值是1，所以先要把坐标减一下
            for index, start in enumerate(starts):
                mask[int(start): int(start + lengths[index])] = class_id
        mask = mask.reshape((1600, 256)).T

        return mask

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        rows = self.csv[self.csv['ImageId'] == img_id]
        mask = self.decode(rows)
        img = cv2.imread(os.path.join(self.root, 'train_images', img_id))
        img, mask = self.transform(img, mask)
        img = self.normalize(img)
        #mask = make_one_hot(mask, self.num_classes)
        #mask = torch.from_numpy(mask).permute(2, 0, 1)
        return img, mask.float()


if __name__ == '__main__':
    import pandas as pd
    import os
    root = 'data/steel'
    train_csv = pd.read_csv(os.path.join(root, 'train.csv'))
    train_dataset = SteelData(root=root, mode='train',
                              csv=train_csv)
    out = train_dataset[0]
    #import pdb; pdb.set_trace()
    print(out[0].shape)
