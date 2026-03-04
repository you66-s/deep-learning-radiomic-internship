import torch.nn as nn
from model_config import ModelConfig

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( ModelConfig.N_Z, ModelConfig.N_GF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ModelConfig.N_GF * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ModelConfig.N_GF * 8, ModelConfig.N_GF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ModelConfig.N_GF * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ModelConfig.N_GF * 4, ModelConfig.N_GF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ModelConfig.N_GF * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ModelConfig.N_GF * 2, ModelConfig.N_GF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ModelConfig.N_GF),
            nn.ReLU(True),
            nn.ConvTranspose2d( ModelConfig.N_GF, ModelConfig.N_C, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(ModelConfig.N_C, ModelConfig.N_DF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(ModelConfig.N_DF, ModelConfig.N_DF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ModelConfig.N_DF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ModelConfig.N_DF * 2, ModelConfig.N_DF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ModelConfig.N_DF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ModelConfig.N_DF * 4, ModelConfig.N_DF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ModelConfig.N_DF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(ModelConfig.N_DF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)