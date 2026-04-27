import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    def __init__(self, num_outputs: int, in_channels=2):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            # input's weights initialization
            self.backbone.features[0][0].weight[:, 0:1, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            nn.init.kaiming_normal_(self.backbone.features[0][0].weight[:, 1:2, :, :])

        # Freeze all backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layers
        for param in self.backbone.features[0][0].parameters():
            param.requires_grad = True
            
        for layer in self.backbone.features[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Replace classifier
        self.backbone.classifier = nn.Identity()

  
        self.regressor = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),       
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)