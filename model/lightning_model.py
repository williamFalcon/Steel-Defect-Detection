from typing import IO

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.utils.metrics import IoU
from torch.optim import AdamW, lr_scheduler

from model.unet_plus import Decoder


class Model(pl.LightningModule):
    def __init__(self,
                 criterion,
                 num_class=5,
                 segmodel='unet',
                 threshold=0.5,
                 lr=1e-3,
                 backbone='resnet34') -> None:
        super(Model, self).__init__()
        self.lr = lr
        self.ious = []
        self.num_class = num_class
        seg_map = {'unet': smp.Unet, 'fpn': smp.FPN, 'psp': smp.PSPNet}
        self.criterion = criterion
        self.threshold = threshold
        self.decoder = seg_map[segmodel](backbone, encoder_weights='imagenet', classes=num_class,
                                         activation=None)

    def forwardd(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.decoder(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True, factor=0.5, eps=1e-6, min_lr=1e-5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.decoder(x)
        val_loss = self.criterion(y_pred, y)
        self.log("val_loss", val_loss)
        iou = IoU(ignore_channels=[0])(y_pred, y)
        self.ious.append(iou.item())
        return val_loss

    def validation_epoch_end(self, *args, **kwargs):
        self.log("val_iou", np.mean(self.ious))
        self.log("val_iou", np.mean(self.ious), prog_bar=True)
        print(f'val iou: {np.mean(self.ious)}')
        self.ious = []

class PlusModel(Model):
    def __init__(self, criterion, encoder='resnet34', lr=1e-3, num_class=5):
        super(PlusModel,self).__init__()
        self.encoder = get_encoder(encoder)
        self.lr = lr
        self.criterion=criterion
        self.decoder = Decoder(filters=self.encoder.out_channels[1:], num_class=num_class)

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.decoder(self.encoder(x))
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.decoder(self.encoder(x))
        val_loss = self.criterion(y_pred, y)
        self.log("val_loss", val_loss)
        iou = IoU(ignore_channels=[0])(y_pred, y)
        self.ious.append(iou.item())
        return val_loss
