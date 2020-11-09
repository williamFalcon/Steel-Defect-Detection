import segmentation_models_pytorch as spm

class Model(object):
    def __init__(self, model_name,class_num,encoder_weights='imagenet'):
        self.model_name = model_name
        self.class_num = class_num
        self.encoder_weights = encoder_weights
    
    def create_model(self):
        model =spm.Unet(self.model_name, encoder_weights=self.encoder_weights, classes=self.class_num,
                                  activation=None)
        return model