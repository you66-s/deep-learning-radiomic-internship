import torch
import torch.nn as nn

class TextureCNN(nn.Module):
    def __init__(self, num_outputs: int, in_channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Keep original resolution (128x128)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            # Block 2: Down to 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            # Block 3: Down to 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            # Block 4: Down to 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )
        
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Dropout(0.6), # Heavy dropout for the massive flattened layer
            nn.Linear(256 * 16 * 16, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        return self.regressor(out)