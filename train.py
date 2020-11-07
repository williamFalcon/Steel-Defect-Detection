from ignite.metrics import Accuracy, mIoU, ConfusionMatrix, Loss
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from torch.optim import Adam
from dataset.dataset import SteelData
from torch.utils.data import DataLoader
import torch
from util.loss import DiceLoss
from argparse import ArgumentParser
from model.unet import Unet

parser = ArgumentParser('Steel Defect')
parser.add_argument('--root', type=str, default='data/steel')
arg = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset = SteelData(root=arg.root, mode='train')
train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True, drop_last=True)
val_dataset = SteelData(root=arg.root, mode='val')
val_loader = DataLoader(val_dataset
                        , num_workers=8, shuffle=True, drop_last=True)
model = Unet()
criterion = DiceLoss()
optim = Adam(model.parameters())
cm = ConfusionMatrix(num_classes=5)
iou_metric = mIoU(cm)
metric = {
    'loss': Loss(criterion),
    'mIOU': iou_metric
}

trainer = create_supervised_trainer(model, optim, criterion, device)
evaluator = create_supervised_evaluator(model, metric, device)


@trainer.on(Events.ITERATION_COMPLETED)
def log_loss(trainer):
    iteration = trainer.state.iteration
    i = iteration % len(train_loader)
    if iteration % 10 == 0:
        print(f"Epoch {trainer.state.epoch} Iteration {i}/{len(train_loader)} Loss :{trainer.state.output}")


@trainer.on(Events.EPOCH_COMPLETED)
def eval_(trainer):
    evaluator.run(val_loader)
    output = evaluator.state.output
    print(f"Loss {output['loss']} mIOU :{output['mIOU']}")


trainer.run(train_loader, max_epochs=100)
