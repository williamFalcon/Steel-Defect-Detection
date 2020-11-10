# Steel-Defect-Detection
Kaggle Steel defect detection

|        | model | lr_method | lr   | mIOU |
| ------ | ----- | --------- | ---- | ---- |
| target | unet  | radam     | 7e-5 | 0.6  |
| 2      | unet  | adamw     | 7e-5 | 0.585 |

model:
resnext50_32x4d
resnet34
se_resnet50
efficientnet-b0

failed:
# ColorJitter(0.4, 0.3, 0.3), Erasing