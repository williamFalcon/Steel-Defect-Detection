# Steel-Defect-Detection
Kaggle Steel defect detection

|     | model    | lr_method | lr   | mIOU  | activation |
| --- | -------- | --------- | ---- | ----- | ---------- |
| 1   | unet++16 | adam      | 6e-4 | 0.423 | HardSwish  |
| 2   | unet++32 | adam      | 6e-4 | 0.44  | HardSwish  |
| 3   | unet++16 | radam     | 6e-4 |  0.44    | HardSwish  |