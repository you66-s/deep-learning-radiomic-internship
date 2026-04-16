import wandb, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, logging
import pandas as pd
from radiomic_dataset import RadiomicDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as T
from architectures.resnet18 import ResNet18
import torch.nn as nn
import torch.optim as optim
from training_engine import custom_scaling_v_hybrid, evaluate_and_plot, train_model, plot_loss_curves, glcm_hybrid_scaler
import numpy as np
from helpers import check_tensor_integrity

# Hyperparameters
tensor_path   = "data/processed_tensors/STAT/224x224_scaled"
LR = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 30
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# W&B parameters
RUN_NAME = f"RESNET-18-no-clipping-{EPOCHS}ep"
RUN_DESCRIPTION = "For this run i removed clipping and i add kurt and skew to quantile features"   
CONFIG = {
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "architecture": "ResNet18",
        "dataset": "2d_stat_1_slice_cleaned_dataset",
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "description": RUN_DESCRIPTION
    }

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# training pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv("data/processed_dataset/STAT/2d_1_slice_statistical_radiomics_dataset_cleaned.csv")
print(f"Training on {len(dataset)} real rows from {dataset['patient_id'].nunique()} patients")
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()
print(f"Training on {len(dataset)} real rows from {dataset['patient_id'].nunique()} patients")

target_cols = [c for c in dataset.columns if c.startswith("stat_")]

unique_patients = dataset['patient_id'].unique()    
train_ids, temp_ids = train_test_split(unique_patients, test_size=0.30, random_state=RANDOM_SEED)   # training set (70%)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=RANDOM_SEED)    # Val (15%) and Test (15%)
train_df = dataset[dataset['patient_id'].isin(train_ids)].copy()
val_df   = dataset[dataset['patient_id'].isin(val_ids)].copy()
test_df  = dataset[dataset['patient_id'].isin(test_ids)].copy()

# saving original indexes
train_df["original_idx"] = train_df.index
val_df["original_idx"]   = val_df.index
test_df["original_idx"]  = test_df.index

logger.info("--- Tensor Alignment Check ---")
check_tensor_integrity(train_df, tensor_path, "TRAIN")
check_tensor_integrity(val_df, tensor_path, "VAL")
check_tensor_integrity(test_df, tensor_path, "TEST")

cols_to_drop = ['patient_id', 'mask_path', 'ct_image_path']
train_df = train_df.drop(labels=cols_to_drop, axis=1)
val_df   = val_df.drop(labels=cols_to_drop, axis=1)
test_df  = test_df.drop(labels=cols_to_drop, axis=1)

train_df, scaler = custom_scaling_v_hybrid(train_df, target_cols, is_train=True)
val_df,  _       = custom_scaling_v_hybrid(val_df,   target_cols, is_train=False, scaler=scaler)
test_df, _       = custom_scaling_v_hybrid(test_df,  target_cols, is_train=False, scaler=scaler)

train_dataset = RadiomicDataset(dataset=train_df, target_cols=target_cols, tensor_dir=tensor_path, is_train=True)
val_dataset   = RadiomicDataset(dataset=val_df, target_cols=target_cols,   tensor_dir=tensor_path, is_train=False)
test_dataset  = RadiomicDataset(dataset=test_df, target_cols=target_cols,  tensor_dir=tensor_path, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=VAL_BATCH_SIZE,   shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=VAL_BATCH_SIZE,   shuffle=False)
logger.info(f"Dataloaders ready: Train={len(train_ids)} pts, Val={len(val_ids)} pts, Test={len(test_ids)} pts")

num_radiomic_features = len([c for c in dataset.columns if c.startswith("stat_")])

model = ResNet18(num_outputs=num_radiomic_features, in_channels=2)

loss_fn = nn.HuberLoss()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS, eta_min=1e-6)

wandb.init(
    project="Encov-Internship",
    name=RUN_NAME,
    config=CONFIG
)
logger.info("Starting training...")
history = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=EPOCHS, device=device, scheduler=scheduler)

model_path = f"artifacts/saved-models/stat-features/{RUN_NAME}.pth"
torch.save(model.state_dict(), model_path)

test_results, fig_eval = evaluate_and_plot(model, test_loader, target_cols, device, RUN_NAME)

wandb.log({
    "test_r2_report": wandb.Image(fig_eval),
    "mean_test_r2": test_results['R2_Score'].mean()
})

fig = plot_loss_curves(history)

artifact = wandb.Artifact(
    name=RUN_NAME,
    type="model",
    description=RUN_DESCRIPTION,
    metadata=CONFIG
)
artifact.add_file(model_path)
wandb.log_artifact(artifact)

wandb.finish()