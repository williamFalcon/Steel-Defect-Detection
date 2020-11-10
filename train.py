import pandas as pd
import torch
from dataset.dataset import SteelData
from torch.utils.data import DataLoader
import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from model.lightning_model import Model
import torch.nn as nn


def create_dataloader(swarg):
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


if __name__ == '__main__':
    parser = ArgumentParser('Steel Defect')
    parser.add_argument('--root', type=str, default='data/steel')
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--group', type=int, default=16, help="Unet groups")
    parser.add_argument('--lr', type=float, default=6e-4, help='defalut lr')
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
