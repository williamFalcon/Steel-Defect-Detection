from typing import List
import numpy as np
from numpy.lib.type_check import imag
import pandas as pd
from tqdm import tqdm


class LabelLoader(object):
    def __init__(self, csv: pd.DataFrame) -> None:
        self.csv = csv
        self.mask_map = {}
        self.w = 1600
        self.h = 256

    def decode(self, row):
        '''
        ImageId,ClassId,EncodedPixels
        '''
        image_id = row['ImageId']
        class_id = row['ClassId']
        if self.mask_map.get(image_id, None) is None:
            self.mask_map[image_id] = np.zeros(self.h*self.w)
        mask = self.mask_map[image_id]
        encoded_pixels = np.array([int(ele)
                                   for ele in row['EncodedPixels'].split(' ')])
        starts, lengths = encoded_pixels[::2], encoded_pixels[1::2]
        starts -= 1  # 因为起始值是1，所以先要把坐标减一下
        ends = starts + lengths
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
        self.mask_map[image_id] = mask

    def get_mask(self, img_id):
        return self.mask_map[img_id].reshape((1600, 256)).T

    def run(self):
        for i in tqdm(range(len(self.csv)), desc="Decode Labels"):
            self.decode(self.csv.iloc[i])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    import random
    import os
    image_path = 'data/steel/train_images'
    csv = pd.read_csv('data/steel/train.csv')
    loader = LabelLoader(csv)
    loader.run()
    key = random.choice(list(loader.mask_map.keys()))
    mask = loader.get_mask(key)
    plt.figure(figsize=(5, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(Image.open(os.path.join(image_path, key)))
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.show()
