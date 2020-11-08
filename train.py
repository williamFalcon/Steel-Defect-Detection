import os
from argparse import ArgumentParser

import pandas as pd
import torch
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import Accuracy, ConfusionMatrix, Loss, mIoU
from numpy.lib.arraysetops import union1d
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader

from dataset.dataset import SteelData
from model.deeplabv3_plus import DeeplabV3Plus
from model.unet_plus import Unet
from util.loss import DiceLoss, lovasz_softmax
from util.lr_scheduler import WarmupMultiStepLR

if not os.path.exists('weights'):
    os.mkdir('weights')

parser = ArgumentParser('Steel Defect')
parser.add_argument('--root', type=str, default='data/steel')
parser.add_argument('--n_cpu', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--group', type=int, default=16, help="Unet groups")
parser.add_argument('--lr', type=float, default=7e-5, help='defalut lr')
arg = parser.parse_args()
print(arg)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
train_csv = pd.read_csv(os.path.join(arg.root, 'train.csv'))
train_dataset = SteelData(root=arg.root, mode='train',
                          csv=train_csv)
train_dataset[0]
train_loader = DataLoader(train_dataset, num_workers=arg.n_cpu,
                          shuffle=True, drop_last=True, batch_size=arg.batch_size)
val_dataset = SteelData(root=arg.root, mode='val',
                        csv=train_csv)
val_loader = DataLoader(val_dataset, num_workers=arg.n_cpu,
                        shuffle=True, drop_last=True, batch_size=arg.batch_size)
model = Unet(5, cc=arg.group).to(device)
bce = BCEWithLogitsLoss().cuda()
dice = DiceLoss().cuda()


def criterion(y_pred, y):
    bce_loss = bce(y_pred, y)
    dice_loss = dice(y_pred, y)
    return bce_loss + dice_loss


optim = AdamW(model.parameters(), lr=arg.lr, weight_decay=4e-5)


def output_transform(output):
    y_pred, y = output
    y = torch.argmax(y, dim=1)
    return (y_pred, y)


cm = ConfusionMatrix(
    num_classes=5, output_transform=output_transform, device=device)
iou_metric = mIoU(cm, ignore_index=0)
metric = {
    'loss': Loss(criterion),
    'mIOU': iou_metric
}

trainer = create_supervised_trainer(
    model, optim, criterion, device)
evaluator = create_supervised_evaluator(model, metric, device)

last_iou = 0


@trainer.on(Events.ITERATION_COMPLETED)
def log_loss(trainer):
    # lr_scheduler.step()
    iteration = trainer.state.iteration
    i = iteration % len(train_loader)
    if iteration % 20 == 0:
        print(
            f"Epoch {trainer.state.epoch} Iteration {i}/{len(train_loader)} Loss :{trainer.state.output}")


@trainer.on(Events.EPOCH_COMPLETED)
def eval_(trainer):
    evaluator.run(val_loader)
    output = evaluator.state.metrics
    global last_iou
    if output['mIOU'] > last_iou:
        last_iou = output['mIOU']
        torch.save(model, f'weights/best_unet++.pth')
    print(">>" * 20)
    with open('out.txt', 'a') as f:
        s = f"Epoch {trainer.state.epoch} Loss {output['loss']} mIOU :{output['mIOU']}"
        print(s)
        f.write(s)
    print(">>"*20)


trainer.run(train_loader, max_epochs=100)
