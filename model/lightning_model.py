from typing import IO
import pytorch_lightning as pl
from segmentation_models_pytorch.utils.metrics import IoU
import segmentation_models_pytorch as smp
from torch.optim import AdamW
import torch


class Model(pl.LightningModule):
    def __init__(self,
                 criterion,
                 num_class=5,
                 segmodel='unet',
                 threshold=0.5,
                 backbone='resnet34') -> None:
        self.num_class = num_class
        seg_map = {'unet': smp.Unet, 'fpn': smp.FPN, 'psp': smp.PSPNet}
        self.criterion = criterion
        self.decoder = seg_map[segmodel](backbone)

    def forwardd(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.decoder(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.decoder(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss)
        y_pred[y_pred > self.threshold] = 1
        pred = torch.argmax(y_pred, dim=1)
        target = torch.argmax(y, dim=1)
        iou = IoU()(pred, target, ignore_index=0, num_classes=5)
        self.log('val_iou', iou)
        return loss
