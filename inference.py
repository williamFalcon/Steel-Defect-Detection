import cv2
import os
from argparse import ArgumentParser
from torchvision.transforms import Compose, ToPILImage, ToTensor, Normalize
import torch
import matplotlib.pyplot as plt
parser = ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--dir', type=str)
arg = parser.parse_args()
transform = Compose([
    ToPILImage(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])])

threshold = 0.7
model = None
plt.figure(figsize=(10, 10))
for x in os.listdir(arg.dir):
    img_id = os.path.join(arg.dir, x)
    img = cv2.imread(img_id)
    left = img[:, :800, :]
    right = img[:, 800:, :]
    left = transform(left)
    right = transform(right)
    left_out = model(left)
    right_out = model(right)
    left_out[left_out > threshold] = 1
    right_out[right_out > threshold] = 1
    left_mask = torch.argmax(left_out, dim=1)
    right_mask = torch.argmax(right_out, dim=1)
    plt.subplot(3, 1, 1)
    plt.imshow(left_mask)
    plt.subplot(3, 1, 2)
    plt.imshow(right_mask)
    plt.subplot(3, 1, 3)
    plt.imshow(img)
