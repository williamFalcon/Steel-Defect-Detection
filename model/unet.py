from torch import nn
import torch
import torch.nn.functional as F
from model.resnet import ResNet34


class UpBlock(nn.Module):
    def __init__(self, first, second, third) -> None:
        super(UpBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.BatchNorm2d(second),
            nn.ReLU(True),
            nn.Conv2d(second, third, kernel_size=3, padding=1),
            nn.BatchNorm2d(third),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class Unet(nn.Module):
    def __init__(self, n_class=4) -> None:
        super(Unet, self).__init__()
        self.classes = n_class
        self.backbone = ResNet34()
        self.trans = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.up1 = UpBlock(512, 256, 128)
        self.up2 = UpBlock(256, 128, 64)
        self.up3 = UpBlock(128, 64, 32)
        self.up4 = UpBlock(64, 32, 32)
        self.classify = nn.Sequential(
            nn.Conv2d(64, n_class, 1)
        )

    def forward(self, x):
        first, second, third, fourth, fifth = self.backbone(x)
        x = self.trans(fifth)
        import pdb
        pdb.set_trace()
        h, w = fourth.shape[2:]
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        
        x = torch.cat([fourth, x],dim=1)

        x = self.up1(x)
        h, w = third.shape[2:]
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        x = torch.cat([third, x],dim=1)
        x = self.up2(x)
        h, w = second.shape[2:]
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        x = torch.cat([second, x],dim=1)
        x = self.up3(x)
        h, w = first.shape[2:]
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        x = torch.cat([first, x],dim=1)
        x = self.up4(x)

        x = self.classify(x)
        return x


if __name__ == '__main__':
    import torch
    model = Unet()
    data = torch.rand((1, 3, 224, 224))
    output = model(data)
    print(output.shape)
