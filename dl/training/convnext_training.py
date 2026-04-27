import wandb, sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, logging
import pandas as pd
from radiomic_dataset import RadiomicDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as T
from architectures.convnext import ConvNeXtRadiomics
import torch.nn as nn
import torch.optim as optim
from training_engine import evaluate_and_plot, train_model, plot_loss_curves, glcm_hybrid_scaler
import numpy as np

# Hyperparameters
LR = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 20
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# W&B parameters
RUN_NAME = f"convnext-glcm-run-{EPOCHS}ep"
RUN_DESCRIPTION = "Switched from ResNet18 to ConvNeXt-Tiny backbone to leverage a larger receptive field (7x7 kernels) and modern training techniques."   
CONFIG = {
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "architecture": "convnext-tiny",
        "dataset": "2d_glcm_cleaned_texture_dataset",
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

dataset = pd.read_csv("data/processed_dataset/glcm_cleaned_texture_dataset.csv")
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()


target_cols = [c for c in dataset.columns if c.startswith("cm_")]

unique_patients = dataset['patient_id'].unique()    
train_ids, temp_ids = train_test_split(unique_patients, test_size=0.30, random_state=RANDOM_SEED)   # training set (70%)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=RANDOM_SEED)    # Val (15%) and Test (15%)
train_df = dataset[dataset['patient_id'].isin(train_ids)].copy()
val_df   = dataset[dataset['patient_id'].isin(val_ids)].copy()
test_df  = dataset[dataset['patient_id'].isin(test_ids)].copy()

cols_to_drop = ['patient_id', 'mask_path', 'ct_image_path']
train_df = train_df.drop(labels=cols_to_drop, axis=1)
val_df   = val_df.drop(labels=cols_to_drop, axis=1)
test_df  = test_df.drop(labels=cols_to_drop, axis=1)


POWER_FEATURES = [
    'cm_joint_avg_d1_2d_avg_fbn_n16',
    'cm_joint_entr_d1_2d_avg_fbn_n16',
    'cm_diff_avg_d1_2d_avg_fbn_n16',
    'cm_diff_entr_d1_2d_avg_fbn_n16',
    'cm_sum_avg_d1_2d_avg_fbn_n16',
    'cm_sum_entr_d1_2d_avg_fbn_n16',
    'cm_inv_diff_d1_2d_avg_fbn_n16',
    'cm_inv_diff_norm_d1_2d_avg_fbn_n16',
    'cm_inv_diff_mom_d1_2d_avg_fbn_n16',
    'cm_inv_diff_mom_norm_d1_2d_avg_fbn_n16',
    'cm_inv_var_d1_2d_avg_fbn_n16',
    'cm_auto_corr_d1_2d_avg_fbn_n16'
]
QUANTILE_FEATURES = [
    'cm_joint_max_d1_2d_avg_fbn_n16',
    'cm_joint_var_d1_2d_avg_fbn_n16',
    'cm_diff_var_d1_2d_avg_fbn_n16',
    'cm_sum_var_d1_2d_avg_fbn_n16',
    'cm_energy_d1_2d_avg_fbn_n16',
    'cm_contrast_d1_2d_avg_fbn_n16',
    'cm_dissimilarity_d1_2d_avg_fbn_n16',
    'cm_corr_d1_2d_avg_fbn_n16',
    'cm_clust_tend_d1_2d_avg_fbn_n16',
    'cm_clust_shade_d1_2d_avg_fbn_n16',
    'cm_clust_prom_d1_2d_avg_fbn_n16',
    'cm_info_corr1_d1_2d_avg_fbn_n16',
    'cm_info_corr2_d1_2d_avg_fbn_n16'
]

train_df, preprocessors     = glcm_hybrid_scaler(train_df, power_features=POWER_FEATURES, quantile_features=QUANTILE_FEATURES, is_train=True)
val_df,  _                  = glcm_hybrid_scaler(val_df,   power_features=POWER_FEATURES, quantile_features=QUANTILE_FEATURES, is_train=False, preprocessors=preprocessors)
test_df, _                  = glcm_hybrid_scaler(test_df,  power_features=POWER_FEATURES, quantile_features=QUANTILE_FEATURES, is_train=False, preprocessors=preprocessors)

tensor_path = "data/processed_tensors/128x128"
train_dataset = RadiomicDataset(dataset=train_df, tensor_dir=tensor_path, target_cols=target_cols, is_train=True)
val_dataset   = RadiomicDataset(dataset=val_df,   tensor_dir=tensor_path, target_cols=target_cols, is_train=False)
test_dataset  = RadiomicDataset(dataset=test_df,  tensor_dir=tensor_path, target_cols=target_cols, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=VAL_BATCH_SIZE,   shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=VAL_BATCH_SIZE,   shuffle=False)
logger.info(f"Dataloaders ready: Train={len(train_ids)} pts, Val={len(val_ids)} pts, Test={len(test_ids)} pts")

num_radiomic_features = len([c for c in dataset.columns if c.startswith("cm_")])

model = ConvNeXtRadiomics(num_outputs=num_radiomic_features, in_channels=2)

loss_fn = nn.HuberLoss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS, eta_min=1e-6)

wandb.init(
    project="GLCM-Expriments",
    name=RUN_NAME,
    config=CONFIG
)
logger.info("Starting training...")
history = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=EPOCHS, device=device, scheduler=scheduler)

model_path = f"artifacts/saved-models/{RUN_NAME}.pth"
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