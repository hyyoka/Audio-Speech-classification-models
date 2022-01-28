from __future__ import print_function 
from __future__ import division

import logging
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class ImageModels():
    def __init__(self,logger: Union[logging.Logger, None] = None) -> None:
        super().__init__(logger=logger)
        self.feature_extract = False 
        
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def observe_parameter(self, model):
        params_to_update = model.parameters()
        # print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        return params_to_update

    def build_model(self, model_config, num_classes) -> torch.nn.Module:
        use_pretrained = True

        if model_config == 'alexnet':
            model = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        elif model_config == 'densenet':
            model = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif model_config == 'inception':
            model = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model, self.feature_extract)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,num_classes)
        elif model_config == 'resnet':
            model = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_config == 'squeezenet':
            model = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model, self.feature_extract)
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model.num_classes = num_classes
        elif model_config == 'vgg19':    
            model = models.vgg19_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        return model
