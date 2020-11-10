import pandas as pd
import torch
<<<<<<< HEAD
from dataset.dataset import SteelData
from torch.utils.data import DataLoader
import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from model.lightning_model import Model
import torch.nn as nn
=======
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import ConfusionMatrix, IoU, Loss, mIoU
from ignite.utils import convert_tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import SteelData
from model.model import Model
from util.loss import DiceLoss, lovasz_softmax
from util.optimizer import RAdam

import numpy as np
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
from util.meter import Meter


class Trainer(object):
    def __init__(self, arg) -> None:
        self.arg = arg
        if not os.path.exists('weights'):
            os.mkdir('weights')

        self.writer = SummaryWriter(log_dir='run')

    def create_dataloader(self, arg):
        train_csv = pd.read_csv(os.path.join(arg.root, 'train.csv'))
        train_dataset = SteelData(root=arg.root, mode='train', csv=train_csv)

        train_loader = DataLoader(train_dataset,
                                  num_workers=arg.n_cpu,
                                  shuffle=True,
                                  drop_last=True,
                                  batch_size=arg.batch_size)
        val_dataset = SteelData(root=arg.root, mode='val', csv=train_csv)
        val_loader = DataLoader(val_dataset,
                                num_workers=arg.n_cpu,
                                shuffle=True,
                                drop_last=True,
                                batch_size=arg.batch_size)
        return train_loader, val_loader

    def run(self):
        train_loader, val_loader = self.create_dataloader(self.arg)
        segmodel = Model(arg.model, 4, device=device).create_model()
        criterion = BCEWithLogitsLoss()
        if self.arg.radam:
            print("Use Radam")
            optim = RAdam(segmodel.parameters(), lr=arg.lr, weight_decay=4e-5)
        else:
            optim = AdamW(segmodel.parameters(), lr=arg.lr, weight_decay=4e-5)

        lr_scheduler = CosineAnnealingLR(optim, 10)
        self.last_iou = 0
        trainer = create_supervised_trainer(segmodel,
                                            optim,
                                            criterion,
                                            device=device)

        def val(epoch, val_loader):
            lr_scheduler.step()
            segmodel.eval()
            meter = Meter()
            losses = []
            for batch in val_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_pred = segmodel(x)
                loss = criterion(y_pred, y)
                losses.append(loss.item())
                meter.update(y.cpu().detach(), y_pred.cpu().detach())
            dices, iou = meter.get_metrics()
            dice, dice_neg, dice_pos = dices
            print(
                f"val loss {round(np.mean(losses),2)} dice:{dice} dice_neg:{dice_neg} dice_pos{dice_pos}")
            print(f"iou : {iou}")
            self.writer.add_scalar('iou', iou, epoch)
            self.writer.add_scalar('val_loss', np.mean(losses), epoch)
>>>>>>> e3f847380a94942a5dfac219f8c77bd0faa406e4


<<<<<<< HEAD
def create_dataloader(swarg):
    train_csv = pd.read_csv(os.path.join(arg.root, 'train.csv'))
    train_dataset = SteelData(root=arg.root, mode='train', csv=train_csv)
=======
        @trainer.on(Events.EPOCH_COMPLETED)
        def eval(trainer):
            val(trainer.state.epoch, val_loader)
>>>>>>> e3f847380a94942a5dfac219f8c77bd0faa406e4

    train_loader = DataLoader(train_dataset,
                            num_workers=arg.n_cpu,
                            shuffle=True,
                            drop_last=True,
                            batch_size=arg.batch_size)
    val_dataset = SteelData(root=arg.root, mode='val', csv=train_csv)
    val_loader = DataLoader(val_dataset,
                            num_workers=arg.n_cpu,
                            shuffle=True,
                            drop_last=True,
                            batch_size=arg.batch_size)
    return train_loader, val_loader


if __name__ == '__main__':
    parser = ArgumentParser('Steel Defect')
    parser.add_argument('--root', type=str, default='data/steel')
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--group', type=int, default=16, help="Unet groups")
    parser.add_argument('--lr', type=float, default=7e-5, help='defalut lr')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model',
                        type=str,
                        default='resnet34',
                        help='efficient net  choose')
    parser.add_argument('--radam', action='store_true')
    arg = parser.parse_args()
    print(arg)
    model = Model(criterion=nn.BCEWithLogitsLoss())
    train_loader, val_loader = create_dataloader()
    trainer = pl.Trainer(model, gpus=1, auto_lr_find=True)
    trainer.fit(train_dataloader=train_loader, val_dataloaders=val_loader)
