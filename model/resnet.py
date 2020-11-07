from typing import Sequence
from pandas.core.algorithms import isin
from torch import nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, s=1, activation='swish') -> None:
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 1, 1)
        )
        self.transform = None
        if s == 2:
            self.transform = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        short = x
        x = self.conv(x)
        if self.transform:
            short = self.transform(short)
        x = short + x
        x = nn.ReLU()(x)
        return x

from torchvision.models import resnet34

class ResNet34(nn.Module):
    def __init__(self) -> None:
        super(ResNet34, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.second = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self.third = nn.Sequential(
            ResidualBlock(64, 128, s=2),
            ResidualBlock(128, 128),
        )

        self.fourth = nn.Sequential(
            ResidualBlock(128, 256, s=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        self.fifth = nn.Sequential(
            ResidualBlock(256, 512,s=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        first = self.first(x)
        second = self.second(first)
        third = self.third(second)
        fourth = self.fourth(third)
        fifth = self.fifth(fourth)
        return [first, second, third, fourth, fifth]


if __name__ == '__main__':
    data = torch.rand((1, 3, 224, 224))
    model = ResNet34()
    output = model(data)
    
