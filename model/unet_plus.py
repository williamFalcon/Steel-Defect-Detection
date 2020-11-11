import torch
import torch.nn.functional as F
from torch import nn

from segmentation_models_pytorch.encoders import get_encoder


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
        self.transform = nn.Sequential([
            ConvBlock(filters[0], filters[1]),
            ConvBlock(filters[1],filters[1])
        ])

    def forward(self, x, shorts):
        shape = shorts[0].shape[2:]
        x = F.interpolate(x, shape, mode='bilinear', align_corners=True)
        shorts.append(x)
        x = torch.cat(shorts, dim=1)
        x = self.transform(x)
        return x


class Decoder(nn.Module):
    def __init__(self, filters, num_class):
        '''
        00-->01-->02-->03-->04
         \  /   /    /    /
         10-->11-->12-->13
          \  /   /    /
          20-->21---22
           \  /    /
           30---->31
            \   /
             40
        '''
        super(Decoder, self).__init__()
        zero, first, second, third, fourth = filters
        self.u01 = UpSample([zero + first, zero])
        self.u02 = UpSample([zero * 2 + first, zero])
        self.u03 = UpSample([zero * 3 + first, zero])
        self.u04 = UpSample([zero * 4 + first, zero])
        self.u11 = UpSample([first + second, first])
        self.u12 = UpSample([first * 2 + second, first])
        self.u13 = UpSample([first * 3 + second, first])
        self.u21 = UpSample([second + third, second])
        self.u22 = UpSample([second * 2 + third, second])
        self.u31 = UpSample([third + fourth, third])
        self.cl1 = ConvBlock(zero, num_class, 1)
        self.cl2 = ConvBlock(zero, num_class, 1)
        self.cl3 = ConvBlock(zero, num_class, 1)
        self.cl4 = ConvBlock(zero, num_class, 1)
        self.deepsupervise = nn.Conv2d(num_class * 4, num_class, 1)

    def forward(self, x):
        H, W = x[0].shape[2:]
        x00, x10, x20, x30, x40 = x[1:]
        x31 = self.u31(x40, [x30])
        x21 = self.u21(x30, [x20])
        x22 = self.u22(x31, [x20, x21])
        x11 = self.u11(x20, [x10])
        x12 = self.u12(x21, [x10, x11])
        x13 = self.u13(x22, [x10, x11, x12])
        x01 = self.u01(x10, [x00])
        x02 = self.u02(x11, [x00, x01])
        x03 = self.u03(x12, [x00, x01, x02])
        x04 = self.u04(x13, [x00, x01, x02, x03])
        u1 = self.cl1(x01)
        u2 = self.cl2(x02)
        u3 = self.cl3(x03)
        u4 = self.cl4(x04)
        uc = torch.cat([u1,u2,u3,u4],dim=1)
        ui = F.interpolate(uc,(H,W),mode='bilinear',align_corners=True)
        uo = self.deepsupervise(ui)
        return  uo


if __name__=='__main__':
    import torch
    inputs = torch.rand((1,3,224,224))
    encoder = get_encoder('resnet34')
    enc_out = encoder(inputs)
    decoder = Decoder(encoder.out_channels[1:],4)
    dec_out = decoder(enc_out)
