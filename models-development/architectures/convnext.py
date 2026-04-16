import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtRadiomics(nn.Module):
    def __init__(self, num_outputs: int = 17, in_channels: int = 6):
        super().__init__()
 
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # 1. Adaptation de la Première couche

        old_stem = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels, 
            old_stem.out_channels, 
            kernel_size=old_stem.kernel_size, 
            stride=old_stem.stride
        )
        
        # Initialisation intelligente des poids pour 10 canaux
        with torch.no_grad():
            # On moyenne les poids RGB pour initialiser les nouveaux canaux
            avg_weight = old_stem.weight.mean(dim=1, keepdim=True) 
            self.backbone.features[0][0].weight = nn.Parameter(
                avg_weight.repeat(1, in_channels, 1, 1)
            )

        # 2. Remplacement de la tête de classification par un régresseur
        # Sur ConvNeXt-Tiny, la sortie avant le classifier est de dimension 768
        self.backbone.classifier[2] = nn.Identity() 

        self.regressor = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512), # ConvNeXt utilise LayerNorm au lieu de BatchNorm
            nn.GELU(),         # ConvNeXt utilise GELU au lieu de ReLU/LeakyReLU
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)