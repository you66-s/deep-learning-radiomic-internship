import wandb, sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, logging
import pandas as pd
from radiomic_dataset import RadiomicDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as T
from architectures.resnet18 import ResNet18
import torch.nn as nn
import torch.optim as optim
from training_engine import train_model, plot_loss_curves
import numpy as np

# Hyperparameters
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# W&B parameters
RUN_NAME = "resnet18-normalized-ct-values-cosine-lr-100ep"
RUN_DESCRIPTION = "xtended training to 100 epochs with CosineAnnealingLR scheduler replacing ReduceLROnPlateau."

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# training pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv("data/processed_dataset/final_2d_cleaned_dataset.csv")
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()


target_cols = [c for c in dataset.columns if c.startswith("stat_")]

unique_patients = dataset['patient_id'].unique()
train_ids, val_ids = train_test_split(unique_patients, test_size=0.2, random_state=RANDOM_SEED)
train_data = dataset[dataset['patient_id'].isin(train_ids)]
val_data   = dataset[dataset['patient_id'].isin(val_ids)]

train_data = train_data.drop(labels=['patient_id', 'mask_path', 'ct_image_path'], axis=1)
val_data = val_data.drop(labels=['patient_id', 'mask_path', 'ct_image_path'], axis=1)
# features normalization
scaler = StandardScaler()
train_data[target_cols] = scaler.fit_transform(train_data[target_cols])
val_data[target_cols] = scaler.transform(val_data[target_cols])

print("train data example: ", train_data.iloc[10].values)

train_dataset = RadiomicDataset(csv_dataset=train_data, tensor_dir="data/processed_tensors/128x128", is_train=True)
val_dataset = RadiomicDataset(csv_dataset=val_data, tensor_dir="data/processed_tensors/128x128", is_train=False)

logger.info("Creating DataLoaders")
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
logger.info("DataLoaders ready")

num_radiomic_features = len([c for c in dataset.columns if c.startswith("stat_")])

logger.info("Initializing ResNet18 model")
model = ResNet18(num_outputs=num_radiomic_features, in_channels=2)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=1e-6)

wandb.init(
    project="Encov-Internship",
    name=RUN_NAME,
    config={
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "architecture": "ResNet18",
        "dataset": "CT-Radiomics-2D-features",
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "description": RUN_DESCRIPTION
    }
)
logger.info("Starting training...")
history = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=EPOCHS, device=device)

fig = plot_loss_curves(history)


wandb.finish()