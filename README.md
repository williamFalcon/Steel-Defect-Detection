# Steel-Defect-Detection
Kaggle Steel defect detection

|        | model | lr_method | lr   | criterion | mIOU |
| ------ | ----- | --------- | ---- | --------- | ---- |
| target | unet  | radam     | 7e-5 | bce       | 0.6  |
| 2      | fpn   | adamW     | 7e-5 | bce+dice  | 0. 62|

model:
resnext50_32x4d
resnet34
se_resnet50
efficientnet-b0

failed:
# ColorJitter(0.4, 0.3, 0.3), Erasing