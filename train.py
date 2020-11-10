import os
from argparse import ArgumentParser

import pandas as pd
import torch
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

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


def argmax(hot):
    b, c, w, h = hot.shape
    mask = torch.zeros((b, w, h))
    for i in range(b):
        for c in range(4):
            mask[i][hot[i][c] == 1] = i+1
    return mask


def output_transform(batch):
    y_pred, y = batch
    y = argmax(y)
    return y_pred, y


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
        cm = ConfusionMatrix(num_classes=4, device=device,
                             output_transform=output_transform)
        iou_metric = mIoU(cm)
        iou = IoU(cm)
        metric = {'loss': Loss(criterion), 'mIOU': iou_metric, 'IOU': iou}
        self.last_iou = 0
        trainer = create_supervised_trainer(segmodel,
                                            optim,
                                            criterion,
                                            device=device)
        evaluator = create_supervised_evaluator(segmodel, metric, device)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_loss(trainer):
            iteration = trainer.state.iteration
            i = iteration % len(train_loader)
            self.writer.add_scalar('train_loss', trainer.state.output,
                                   iteration)
            if iteration % 20 == 0:
                print(
                    f"Epoch {trainer.state.epoch} Iteration {i}/{len(train_loader)} Loss :{trainer.state.output}"
                )

        @trainer.on(Events.EPOCH_STARTED)
        def eval_(trainer):
            lr_scheduler.step()
            evaluator.run(val_loader)
            output = evaluator.state.metrics
            if output['mIOU'] > self.last_iou:
                self.last_iou = output['mIOU']
                torch.save(segmodel, f'weights/best_unet++.pth')
            self.writer.add_scalar('val_loss', output['loss'],
                                   trainer.state.epoch)
            self.writer.add_scalar('val_miou', output['mIOU'],
                                   trainer.state.epoch)
            print(">>" * 20)
            with open('out.txt', 'a') as f:
                s = f"Epoch {trainer.state.epoch} Loss {output['loss']} mIOU :{output['mIOU']} IOU:{output['IOU']}\n"
                print(s)
                f.write(s)
            print(">>" * 20)

        trainer.run(train_loader, max_epochs=self.arg.epochs)


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
    trainer = Trainer(arg)
    trainer.run()
