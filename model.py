import torch.nn as nn
import torch
from torchvision import models

def build_model(num_classes, pretrained=True):
    # تحميل DenseNet121
    model = models.densenet121(pretrained=pretrained)

    # استبدال الـ classifier (fc layer)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model
