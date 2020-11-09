from segmentation.unet import Decoder
from backbone.efficientnet import  EfficientNet
import torch

inputs = torch.rand((1,3,224,224))
backbone = EfficientNet.from_name('efficientnet-b0')
backbone.eval()
endpoints = backbone.extract_endpoints(inputs)
filters = [endpoints[f'reduction_{i}'].shape[1] for i in range(1,6)]
backbone.train()

unet = Decoder(backbone,filters=filters,num_class=4)
output = unet(inputs)
print(output.shape)