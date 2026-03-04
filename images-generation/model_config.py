import torch
import torch.nn as nn

class ModelConfig:
    BATCH_SIZE = 16    
    IMAGE_SIZE = 64     
    N_C = 1             # Number of channels in the training images.
    N_Z = 100           # Size of z latent vector
    N_GF = 64           # Size of feature maps in generator
    N_DF = 32           # Size of feature maps in discriminator
    LR = 0.0002         # Learning rate for optimizers
    NUM_EPOCHS = 200    # Number of training epochs
    BETA_1 = 0.5        # Beta1 hyperparameter for Adam optimizers
    N_GPU = 1           # Number of GPUs available. Use 0 for CPU mode.
    
    def weights_init(m):
        # mean = 0, stddev= 0.02
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            