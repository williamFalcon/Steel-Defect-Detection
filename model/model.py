from segmentation_models_pytorch.utils.metrics import Accuracy
from typing import Optional
from typing import IO

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.utils.metrics import IoU
from torch.optim import AdamW, lr_scheduler, SparseAdam
from segmentation_models_pytorch.utils.functional import _threshold
from .unet_plus import UnetPP


def cal_dice(pred, target):
    with torch.no_grad():
        pred = torch.sigmoid(pred[:, 1:])
        target = target[:, 1:]
        batch = len(pred)
        pred = _threshold(pred, 0.5)
        p = pred.view(batch, -1)
        t = target.view(batch, -1)
        intersection = (p * t).sum(-1)
        union = (p + t).sum(-1)
        dice = ((2 * intersection) / (union+1e-5)).mean().item()
        return dice


class Model(pl.LightningModule):
    def __init__(self,
                 criterion,
                 num_class=5,
                 decoder='unet',
                 threshold=0.5,
                 lr=1e-3,
                 encoder='resnet34') -> None:
        super(Model, self).__init__()
        self.best_iou = 0
        self.best_dice = 0
        self.lr = lr
        self.dices = []
        self.ious = []
        self.num_class = num_class
        decoder_map = {'unet': smp.Unet, 'fpn': smp.FPN,
                       'psp': smp.PSPNet, 'unetpp': UnetPP}
        self.criterion = criterion
        self.iou_cal = IoU(ignore_channels=[0])
        self.threshold = threshold.as_integer_ratio()
        self.decoder = decoder_map[decoder](encoder, encoder_weights='imagenet', classes=num_class,
                                            activation=None)

    def forwardd(self, x):
        if self.mode == 'segment':
            return self.decoder(x)
        else:
            return self.decoder(x)[1]

    def training_step(self, batch, batch_idx):
        x, mask, _ = batch
        y_pred = self.decoder(x)
        loss = self.criterion(y_pred, mask)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.decoder.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, verbose=True, factor=0.5, eps=1e-6, min_lr=1e-5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

    def validation_step(self, batch, batch_idx):
        x, mask, _ = batch
        y_pred = self.decoder(x)
        loss = self.criterion(y_pred, mask)
        iou = self.iou_cal(y_pred, mask)
        self.dices.append(cal_dice(y_pred, mask))
        self.ious.append(iou.item())
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, *args, **kwargs):
        self.log("iou", np.mean(self.ious))
        self.log("dice", np.mean(self.dices))
        mean_iou = np.mean(self.ious)
        mean_dice = np.mean(self.dices)
        if mean_iou > self.best_iou:
            self.best_iou = mean_iou
            print(f'better iou: {self.best_iou}')
        if mean_dice > self.best_dice:
            self.best_dice = mean_dice
            print(f'better dice: {self.best_dice}')
        self.log("iou", mean_iou, prog_bar=True)
        self.ious = []
        self.dices = []
