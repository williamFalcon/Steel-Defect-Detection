import torch
from ignite.metrics import mIoU,ConfusionMatrix,IoU
from ignite.engine import create_supervised_evaluator
from torch.utils.data import DataLoader, Dataset
from torch import nn


def make_one_hot(labels, num_classes):
    labels = labels.unsqueeze(0).unsqueeze(0)
    one_hot = torch.FloatTensor(labels.size(0), num_classes, 4,4).zero_()
    target = one_hot.scatter(1, labels.data, 1)
    return target


class Data(Dataset):
    def __init__(self):
        super(Data, self).__init__()

    def __len__(self):
        return 1

    def __getitem__(self, item):
        pred = torch.LongTensor([[1, 1, 0, 1],
                                 [3, 2, 1, 1],
                                 [3, 2, 1, 2],
                                 [1, 1, 0, 0]])
        mask = torch.LongTensor([[1, 1, 0, 1],
                                 [1, 2, 1, 0],
                                 [3, 1, 2, 2],
                                 [3, 1, 0, 0]])
        pred = make_one_hot(pred, 4)
        return pred[0], mask

model = nn.Sequential()
criterion =nn.BCELoss()
from torch.optim import Adam
device = torch.device('cpu')
cm = ConfusionMatrix(num_classes=4)
miou=mIoU(cm, ignore_index=0)
iou = IoU(cm,ignore_index=0)
metric = {
    'mIOU':miou,
    'IOU':iou
}
evaluator =create_supervised_evaluator (model,metric,device=device)
data_loader = DataLoader(Data())
evaluator.run(data_loader)
print(evaluator.state.output)



state = evaluator.run(data_loader)
print(state.metrics['mIOU'])
print(state.metrics['IOU'])
