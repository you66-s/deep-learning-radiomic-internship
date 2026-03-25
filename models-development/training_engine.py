import torch, time, logging
from tqdm.auto import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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