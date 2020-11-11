from typing import IO

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.utils.metrics import IoU
from torch.optim import AdamW, lr_scheduler

from .unet_plus import UnetPP


class Model(pl.LightningModule):
    def __init__(self,
                 seg_criterion,
                 cls_criterion,
                 num_class=5,
                 decoder='unet',
                 threshold=0.5,
                 lr=1e-3,
                 encoder='resnet34') -> None:
        super(Model, self).__init__()
        self.lr = lr
        self.ious = []
        self.num_class = num_class
        decoder_map = {'unet': smp.Unet, 'fpn': smp.FPN,
                       'psp': smp.PSPNet, 'unetpp': UnetPP}
        aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=num_class,                 # define number of output labels
        )
        self.seg_criterion = seg_criterion
        self.cls_criterion = cls_criterion
        self.threshold = threshold
        self.decoder = decoder_map[decoder](encoder, encoder_weights='imagenet', classes=num_class,
                                            activation=None, aux_params=aux_params)

    def forwardd(self, x):
        if self.mode == 'segment':
            return self.decoder(x)
        else:
            return self.decoder(x)[1]

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        y_pred, label = self.decoder(x)
        mask_loss = self.seg_criterion(y_pred, mask)
        label_loss = self.cls_criterion(label, y)
        self.log("train_label_loss", 0.4*label_loss)
        self.log("train_mask_loss", 0.6*mask_loss)
        loss = 0.4 * label_loss + 0.6 * mask_loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, verbose=True, factor=0.8, eps=1e-7, min_lr=7e-6)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        y_pred, label = self.decoder(x)
        val_loss = self.criterion(y_pred, mask)
        label_loss = self.cls_criterion(label, y)
        mask_loss = self.seg_criterion(y_pred, mask)
        self.log("val_label_loss", 0.4*label_loss)
        self.log("val_mask_loss", 0.6*mask_loss)
        iou = IoU(ignore_channels=[0])(y_pred, mask)
        self.ious.append(iou.item())
        val_loss = 0.2*label_loss + 0.6*mask_loss
        self.log("val_loss", val_loss)
        return val_loss

    def validation_epoch_end(self, *args, **kwargs):
        self.log("val_iou", np.mean(self.ious))
        self.log("val_iou", np.mean(self.ious), prog_bar=True)
        print(f'val iou: {np.mean(self.ious)}')
        self.ious = []
