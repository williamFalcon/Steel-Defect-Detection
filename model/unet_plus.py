import torch
import torch.nn.functional as F
from torch import nn
import segmentation_models_pytorch as spm
from segmentation_models_pytorch.encoders import get_encoder

import pytorch_lightning as pl


class ConvBlock(nn.Module):
    def __init__(self, inc, ouc, k=3, s=1, p=1):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, ouc, k, s, p),
            nn.BatchNorm2d(ouc),
            nn.Hardswish()
        )

    def forward(self, x):
        return self.net(x)


class UpSample(nn.Module):
    def __init__(self, filters):
        super(UpSample, self).__init__()
        self.transform = nn.Sequential(*[
            ConvBlock(filters[i], filters[i + 1]) for i in range(2)
        ])

    def forward(self, x,  shorts):
        shape = shorts[0].shape[2:]
        x = F.interpolate(x, shape, mode='bilinear', align_corners=True)
        shorts.append(x)
        x = torch.cat(shorts, dim=1)
        x = self.transform(x)
        return x


class Decoder(nn.Module):
    def __init__(self,  filters, num_class):
        '''
        efficent b0 16 24 40 112 1280
        :param filters:
        :param num_class:
        '''
        super(Decoder, self).__init__()
        first, second, third, fourth, fifth = filters
        self.transform = ConvBlock(fifth, fourth)
        self.u31 = UpSample([fourth*2, fourth, third])
        self.u22 = UpSample([third*2, third, second])
        self.u13 = UpSample([second*2, second, first])
        self.u04 = UpSample([first*2, first, first])
        self.classify = nn.Sequential(
            nn.Conv2d(first, num_class, kernel_size=1),
        )

    def forward(self, x):
        H, W = x.shape[2:]
        endpoints = self.backbone.extract_endpoints(x)
        [x00, x10, x20, x30, x40] = [
            endpoints[f'reduction_{i}'] for i in range(1, 6)]
        trans = self.transform(x40)
        x31 = self.u31(trans, [x30])
        x22 = self.u22(x31, [x20])
        x13 = self.u13(x22, [x10])
        x04 = self.u04(x13, [x00])
        x = F.interpolate(x04, (H, W), mode='bilinear', align_corners=True)
        x = self.classify(x)
        return x
