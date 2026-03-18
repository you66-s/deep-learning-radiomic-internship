import sys, os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, logging
import pandas as pd
from radiomic_dataset import RadiomicDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms as T
from architectures.resnet18 import ResNet18
import torch.nn as nn
import torch.optim as optim
from training_engine import train_model, plot_loss_curves
import wandb

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# training pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv("data/final_cleaned_dataset.csv")
dataset = dataset.drop(labels=['patient_id', 'organ', 'is_augmented', 'mask_path', 'ct_image_path'], axis=1)
target_cols = [c for c in dataset.columns if c.startswith("stat_")]

train_data = dataset.sample(frac=0.8, random_state=42)
val_data = dataset.drop(train_data.index)

# features normalization
scaler = StandardScaler()
train_data[target_cols] = scaler.fit_transform(train_data[target_cols])
val_data[target_cols] = scaler.transform(val_data[target_cols])

print("train data example: ", train_data.iloc[10].values)

train_dataset = RadiomicDataset(csv_dataset=train_data, tensor_dir="data/processed_tensors")
val_dataset = RadiomicDataset(csv_dataset=val_data, tensor_dir="data/processed_tensors")

logger.info("Creating DataLoaders")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
logger.info("DataLoaders ready")

num_radiomic_features = len([c for c in dataset.columns if c.startswith("stat_")])

logger.info("Initializing ResNet18 model")
model = ResNet18(num_outputs=num_radiomic_features, in_channels=2)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=2, factor=0.1)

wandb.init(
    project="Encov-Internship",
    name="resnet-Transfert-learning-update-val-batch-size-epochs",
    config={
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "architecture": "ResNet18",
        "dataset": "CT-Radiomics-66k",
        "epochs": 50,
        "train_batch_size": 32,
        "val_batch_size": 128,
        "description": "In this run i modified the batch size of validation set to see its effect and augmente number of epochs to 50"
    }
)
logger.info("Starting training...")
history = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=50, device=device, scheduler=scheduler)

plot_loss_curves(results=history)