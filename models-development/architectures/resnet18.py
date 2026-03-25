import torch, torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, num_outputs: int, in_channels=2):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
            padding=old_conv.padding, bias=False
        )
        
        with torch.no_grad():
            self.backbone.conv1.weight[:, 0:1, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            nn.init.kaiming_normal_(self.backbone.conv1.weight[:, 1:2, :, :])

        # layers to train
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.conv1.parameters():
            param.requires_grad = True

        self.backbone.fc = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),  
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_outputs)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)