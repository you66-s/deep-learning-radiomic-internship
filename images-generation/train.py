import os, mlflow
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model_config import ModelConfig
from dcgan_model_architecture import Generator, Discriminator
from medical_dataset import MedicalCTDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from mlflow_client import MLflowTracker


tracker = MLflowTracker(experiment_name="Medical_DCGAN", run_name="200_EPOCHS_LR_0.0002_Standard")
tracker.log_config(ModelConfig)

# Dataset preparation and loading
# Pass the configured image size so that the data matches the network expectations.
dataset = MedicalCTDataset(image_size=ModelConfig.IMAGE_SIZE)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=ModelConfig.BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)
# Images from dataset:
real_batch = next(iter(dataloader))
tracker.log_image_grid(real_batch[:64], "Real CT Slices", step=0)   # initial real batch sample

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training CT Slices - LR = 0.0002")
grid = vutils.make_grid(real_batch[:64], padding=2, normalize=True)
plt.savefig("images-generation/training_results/training_ct_slices_uplr.png", bbox_inches="tight")
plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create and initialise Generator
netG = Generator(ModelConfig.N_GPU).to(device)
# Handle multi-GPU
if (device.type == 'cuda') and (ModelConfig.N_GPU > 1):
    netG = nn.DataParallel(netG, list(range(ModelConfig.N_GPU)))

# Apply the weights_init function to randomly initialize all weights
netG.apply(ModelConfig.weights_init)

# Create and initialise Generator
netD = Discriminator(ModelConfig.N_GPU).to(device)

if (device.type == 'cuda') and (ModelConfig.N_GPU > 1):
    netD = nn.DataParallel(netD, list(range(ModelConfig.N_GPU)))

netD.apply(ModelConfig.weights_init)

# Print models
print("Discriminator network \n", netD)
print("generator network \n", netG)


# Loss Functions and Optimizers
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, ModelConfig.N_Z, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=ModelConfig.LR, betas=(ModelConfig.BETA_1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=ModelConfig.LR, betas=(ModelConfig.BETA_1, 0.999))

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(ModelConfig.NUM_EPOCHS):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, ModelConfig.N_Z, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # Update G network: maximize log(D(G(z))
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print(f"EPOCH: {epoch}")
            tracker.log_metrics({
                "loss_D": errD.item(),
                "loss_G": errG.item(),
            }, step=iters)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == ModelConfig.NUM_EPOCHS-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                tracker.log_image_grid(fake, f"Fake Slices Epoch {epoch}", step=iters)
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
tracker.log_models(netG, netD)
torch.save(netG.state_dict(), "images-generation/models/generator_LR_0002.pth")
torch.save(netD.state_dict(), "images-generation/models/discriminator_LR_0002.pth")
print("Modèles sauvegardés !")        
# Results 
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training - LR = 0.0002")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("images-generation/training_results/loss_curve_LR_0002.png", bbox_inches="tight")
mlflow.log_artifact("final_loss_curve.png")
tracker.end_run()
print("Training Complete and Logged to MLflow!")
plt.close()