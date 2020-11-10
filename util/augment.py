from math import sqrt

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmenters import segmentation
from imgaug.augmenters.arithmetic import Multiply
from imgaug.augmenters.meta import OneOf


class Augmentor(object):
    def __init__(self, width, height=256):
        self.width = width
        self.height = height

    def aug_train(self, image, segmap):
        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
        # Augment images and segmaps.
        seq = iaa.Sequential(
            [
                iaa.OneOf(
                    [iaa.Sharpen((0.1, 0.3)),  # sharpen the image
                     iaa.AdditiveGaussianNoise(random_state=3),
                     ]
                ),
                iaa.Affine(
                    rotate=(-10, 15)
                ),  # rotate by -45 to 45 degrees (affects segmaps)
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.CropToFixedSize(self.width, self.height)
            ],
            random_order=True,
        )
        image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
        return image_aug, segmap_aug.arr[:, :, 0]

    def aug_val(self, image, segmap):
        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
        # Augment images and segmaps.
        seq = iaa.Sequential(
            [

                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.CropToFixedSize(self.width, self.height)
            ]
        )
        image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
        return image_aug, segmap_aug.arr[:, :, 0]
