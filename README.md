# Steel-Defect-Detection
Kaggle Steel defect detection

|        | model | lr_method | lr   | criterion | mIOU |
| ------ | ----- | --------- | ---- | --------- | ---- |
| target | unet  | radam     | 7e-5 | bce       | 0.6  |
| 2      | unet  | adamw     | 7e-5 | bce       | 0.56 |
| 3      | fpn   | adamW     | 7e-5 | bce       | 0.58 |

model:
resnext50_32x4d
resnet34
se_resnet50
efficientnet-b0

failed:
# ColorJitter(0.4, 0.3, 0.3), Erasing