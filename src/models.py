import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import timm

def get_cnn_model(num_classes):
    # Sử dụng ResNet-50 làm baseline CNN
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_vit_model(num_classes):
    # Sử dụng ViT base patch 16
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)
    return model