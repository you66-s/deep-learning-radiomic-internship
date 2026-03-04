import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model_config import ModelConfig
from model_architecture import Generator

os.makedirs("images-generation/results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le générateur sauvegardé
netG = Generator(ModelConfig.N_GPU).to(device)
netG.load_state_dict(torch.load("images-generation/models/generator_200.pth", map_location=device))
netG.eval()
print("Générateur chargé avec succès !")

# Générer des images
with torch.no_grad():
    noise = torch.randn(64, ModelConfig.N_Z, 1, 1, device=device)
    generated = netG(noise).detach().cpu()

# Afficher et sauvegarder
grid = vutils.make_grid(generated, padding=2, normalize=True)
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Images CT Synthétiques")
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
plt.savefig("images-generation/results/generated_images_LR_0002.png", bbox_inches="tight")
plt.show()

# Sauvegarder individuellement
# for idx in range(len(generated)):
#     img = generated[idx].squeeze().numpy()
#     img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
#     cv2.imwrite(f"images-generation/results/fake_{idx:03d}.png", img)

# print(f"✓ {len(generated)} images sauvegardées dans images-generation/results/")