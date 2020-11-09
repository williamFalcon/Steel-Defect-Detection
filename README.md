# Steel-Defect-Detection
Kaggle Steel defect detection

|     | model    | lr_method | lr   | mIOU  | activation |
| --- | -------- | --------- | ---- | ----- | ---------- |
| 1   | unet++16 | adam      | 6e-4 | 0.423 | HardSwish  |
| 2   | unet++32 | adam      | 6e-4 | 0.44  | HardSwish  |

model:
resnext50_32x4d
resnet34
se_resnet50
efficientnet-b0