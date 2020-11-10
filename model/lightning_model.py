from typing import IO
import pytorch_lightning as pl
from segmentation_models_pytorch.utils.metrics import IoU
import segmentation_models_pytorch as smp
from torch.optim import AdamW, lr_scheduler
import torch
import numpy as np


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
        scheduler = lr_scheduler.CyclicLR(
            optimizer, base_lr=6e-4, max_lr=1e-3, step_size_down=300, step_size_up=300, cycle_momentum=False)
        return [optimizer], [scheduler]

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
        self.log("val_iou", np.mean(self.ious),prog_bar=True)
        print(f'val iou: {np.mean(self.ious)}')
        self.ious = []
