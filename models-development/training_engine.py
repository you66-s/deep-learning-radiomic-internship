import torch, time, logging
from tqdm.auto import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.metrics import r2_score
import wandb

logger = logging.getLogger(__name__)

def train_one_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> float:

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, epochs: int, device: torch.device, scheduler=None) -> Dict[str, List[float]]:

    history = {"train_loss": [], "val_loss": []}
    model.to(device)

    total_start = time.time()

    for epoch in tqdm(range(epochs), desc="Epochs"):

        logger.info(f"Starting epoch {epoch+1}/{epochs}")

        epoch_start = time.time()

        train_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_time = time.time() - train_start

        logger.info(f"Epoch {epoch+1}: training completed in {train_time:.2f}s")

        val_start = time.time()
        val_loss = evaluate(model, val_loader, loss_fn, device)
        val_time = time.time() - val_start
        
        current_lr = optimizer.param_groups[0]['lr']
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr
        })
        
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}: Current Learning Rate: {current_lr:.6f}")
            
        logger.info(f"Epoch {epoch+1}: validation completed in {val_time:.2f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Epoch Time: {epoch_time:.2f}s"
        )
    
    
    total_time = time.time() - total_start
    logger.info(f"Training finished in {total_time:.2f} seconds")

    return history

def evaluate_and_plot(model, loader, target_names, device, run_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("Final Evaluation: Running inference on Test Set...")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    # Calculate R2 for every individual feature
    r2_values = r2_score(y_true, y_pred, multioutput='raw_values')
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Feature': target_names,
        'R2_Score': r2_values
    }).sort_values(by='R2_Score', ascending=False)

    # Print Report
    print(f"\n--- Final Test Performance: {run_name} ---")
    print(results_df.to_string(index=False))

    # Generate Performance Graph
    plt.figure(figsize=(12, 10))
    sns.barplot(data=results_df, x='R2_Score', y='Feature', palette='viridis')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5) # Baseline
    plt.axvline(x=0.8, color='green', linestyle=':', label='High Mastery (0.8)')
    plt.title(f'R² Score per Radiomic Feature (Test Set)\nRun: {run_name}')
    plt.xlabel('R² Score (Higher is Better)')
    plt.tight_layout()
    
    plot_path = f"artifacts/plots/{run_name}_r2_report.png"
    plt.savefig(plot_path)
    return results_df, plt.gcf()

def plot_loss_curves(results):
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]
    epochs = range(1, len(train_loss)+1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()
    

    
def custom_scaling_v3(dataset, target_cols, is_train, scaler=None):
    df = dataset.copy()

    if is_train:
        scaler = {}
        
        # 1. Clipping pour neutraliser les artefacts extrêmes
        clip_bounds = {}
        for col in target_cols:
            low, high = df[col].quantile(0.01), df[col].quantile(0.99)
            clip_bounds[col] = (low, high)
            df[col] = df[col].clip(low, high)
            
        # 2. Yeo-Johnson pour normaliser les distributions
        # method='yeo-johnson' gère les valeurs positives et nulles
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        df[target_cols] = pt.fit_transform(df[target_cols])
        
        # 3. RobustScaler pour le centrage final
        sc = RobustScaler()
        df[target_cols] = sc.fit_transform(df[target_cols])
        
        # On stocke tout dans un dictionnaire unique pour l'inférence
        scaler["unified"] = (sc, target_cols, clip_bounds, pt)
        return df, scaler

    else:
        assert scaler is not None
        sc, cols, clip_bounds, pt = scaler["unified"]
        valid = [c for c in cols if c in df.columns]
        
        # Application des mêmes bornes de clipping du train
        for col in valid:
            if col in clip_bounds:
                low, high = clip_bounds[col]
                df[col] = df[col].clip(low, high)
        
        # Application de la transformation de puissance apprise
        df[valid] = pt.transform(df[valid])
        
        # Application du scaling robuste appris
        df[valid] = sc.transform(df[valid])

        return df, None

def basic_standrdScaler_normalization(dataset, target_cols, is_train, scaler = None):
    df = dataset.copy()
    if is_train:
        scaler = StandardScaler()
        df[target_cols] = scaler.fit_transform(df[target_cols])
        return df, scaler
    else:
        if scaler is None:
            raise ValueError("Scaler not fitted ")
        df[target_cols] = scaler.transform(df[target_cols])
        return df
    
def glcm_hybrid_scaler(dataset, power_features, quantile_features, is_train=True, preprocessors=None):
    df = dataset.copy()

    if set(power_features).intersection(set(quantile_features)):
        raise ValueError("power_features and quantile_features must not overlap")

    all_cols = list(set(power_features + quantile_features))

    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    df[all_cols] = df[all_cols].replace([np.inf, -np.inf], np.nan)

    if is_train:
        medians = df[all_cols].median()
    else:
        if preprocessors is None:
            raise ValueError("Preprocessors must be provided for inference")
        medians = preprocessors["medians"]

    df[all_cols] = df[all_cols].fillna(medians)

    if is_train:
        preprocessors = {}

        if len(power_features) > 0:
            pt_yeo = PowerTransformer(method='yeo-johnson')
            scaler_power = RobustScaler()

            power_data = pt_yeo.fit_transform(df[power_features])
            df[power_features] = scaler_power.fit_transform(power_data)

            preprocessors["pt_yeo"] = pt_yeo
            preprocessors["scaler_power"] = scaler_power

        if len(quantile_features) > 0:
            qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=100)
            scaler_quantile = RobustScaler()

            quantile_data = qt.fit_transform(df[quantile_features])
            df[quantile_features] = scaler_quantile.fit_transform(quantile_data)

            preprocessors["qt"] = qt
            preprocessors["scaler_quantile"] = scaler_quantile

        preprocessors["medians"] = medians
        return df, preprocessors

    else:
        if len(power_features) > 0:
            pt_yeo = preprocessors["pt_yeo"]
            scaler_power = preprocessors["scaler_power"]

            power_data = pt_yeo.transform(df[power_features])
            df[power_features] = scaler_power.transform(power_data)

        if len(quantile_features) > 0:
            qt = preprocessors["qt"]
            scaler_quantile = preprocessors["scaler_quantile"]

            quantile_data = qt.transform(df[quantile_features])
            df[quantile_features] = scaler_quantile.transform(quantile_data)

        return df
    
def custom_scaling_v_hybrid(dataset, target_cols, is_train, scaler=None):
    df = dataset.copy()
    # Features with extreme spike distributions — QuantileTransformer works best
    QUANTILE_FEATURES = ["stat_cov", "stat_qcod"]
    POWER_FEATURES = [c for c in target_cols if c not in QUANTILE_FEATURES]

    if is_train:
        scaler = {}
        clip_bounds = {}

        # Clip outliers for ALL features
        # More aggressive clipping for ratio features
        for col in target_cols:
            if col in QUANTILE_FEATURES:
                low, high = df[col].quantile(0.02), df[col].quantile(0.98)
            else:
                low, high = df[col].quantile(0.01), df[col].quantile(0.99)
            clip_bounds[col] = (low, high)
            df[col] = df[col].clip(low, high)

        # PowerTransformer (Yeo-Johnson) for normal features
        if POWER_FEATURES:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            df[POWER_FEATURES] = pt.fit_transform(df[POWER_FEATURES])
            sc_power = RobustScaler()
            df[POWER_FEATURES] = sc_power.fit_transform(df[POWER_FEATURES])
            scaler["power"] = (pt, sc_power, POWER_FEATURES)

        # QuantileTransformer for spike-distribution ratio features
        if QUANTILE_FEATURES:
            valid_q = [c for c in QUANTILE_FEATURES if c in df.columns]
            qt = QuantileTransformer(
                n_quantiles=1000,
                output_distribution='normal',
                random_state=42
            )
            df[valid_q] = qt.fit_transform(df[valid_q])
            sc_quant = RobustScaler()
            df[valid_q] = sc_quant.fit_transform(df[valid_q])
            scaler["quantile"] = (qt, sc_quant, valid_q)

        scaler["clip_bounds"] = clip_bounds
        return df, scaler

    else:
        assert scaler is not None
        clip_bounds = scaler["clip_bounds"]

        # Apply clip bounds from training
        for col in [c for c in target_cols if c in df.columns]:
            if col in clip_bounds:
                low, high = clip_bounds[col]
                df[col] = df[col].clip(low, high)

        # Apply PowerTransformer pipeline
        if "power" in scaler:
            pt, sc_power, cols = scaler["power"]
            valid = [c for c in cols if c in df.columns]
            df[valid] = pt.transform(df[valid])
            df[valid] = sc_power.transform(df[valid])

        # Apply QuantileTransformer pipeline
        if "quantile" in scaler:
            qt, sc_quant, cols = scaler["quantile"]
            valid = [c for c in cols if c in df.columns]
            df[valid] = qt.transform(df[valid])
            df[valid] = sc_quant.transform(df[valid])

        return df, None