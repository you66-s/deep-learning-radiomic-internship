import wandb, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, logging
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from training_engine import evaluate_and_plot, train_model, plot_loss_curves
from radiomic_dataset import RadiomicDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class Trainer:
    """
    Reusable training pipeline for radiomic feature prediction.
    Swap any model by passing a different model_class at instantiation.

    Usage:
        trainer = RadiomicTrainer(
            model_class=EfficientNetB0,
            model_name="efficientnet-b0",
            architecture="EfficientNet-B0",
        )
        trainer.setup()
        trainer.run()
    """

    # Config
    DEFAULT_CONFIG = {
        "learning_rate":    1e-4,
        "weight_decay":     1e-4,
        "epochs":           30,
        "train_batch_size": 32,
        "val_batch_size":   32,
        "random_seed":      42,
        "in_channels":      2,
        "tensor_dir":       "data/processed_tensors/",
        "csv_path":         "data/processed_dataset/",
        "wandb_project":    "Encov-Internship",
        "cols_to_drop":     ["patient_id", "mask_path", "ct_image_path"],
        "target_prefix":    "stat_",
    }

    def __init__(self, model_class, model_name: str, architecture: str, run_name: str, description: str, **hparams):
        """
        Args:
            model_class:  The model class to instantiate (not an instance, class itself).
            model_name:   Short name used in run names and file paths.
            architecture: architecture name for W&B logging.
            **hparams:    Override any key from DEFAULT_CONFIG.
        """
        self.model_class  = model_class
        self.model_name   = model_name
        self.architecture = architecture

        # Merge defaults with any overrides
        self.cfg = {**self.DEFAULT_CONFIG, **hparams}

        self.run_name = (
            f"{self.model_name}-{run_name}-{self.cfg['epochs']}ep"
        )
        self.run_description = (
            f"{self.architecture} - {description} - "
            f"{self.cfg['epochs']} epochs."
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialized by the setup method
        self.train_loader     = None
        self.val_loader       = None
        self.test_loader      = None
        self.target_cols      = None
        self.model            = None
        self.optimizer        = None
        self.scheduler        = None
        self.loss_fn          = None
        self.history          = None

    # helpers 
    def _set_seed(self):
        seed = self.cfg["random_seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _prepare_split(self, df: pd.DataFrame, ids) -> pd.DataFrame:
        split = df[df["patient_id"].isin(ids)].copy()
        return split.drop(labels=self.cfg["cols_to_drop"], axis=1)

    def _build_wandb_config(self) -> dict:
        return {
            "model":            self.model_name,
            "architecture":     self.architecture,
            "learning_rate":    self.cfg["learning_rate"],
            "weight_decay":     self.cfg["weight_decay"],
            "epochs":           self.cfg["epochs"],
            "train_batch_size": self.cfg["train_batch_size"],
            "val_batch_size":   self.cfg["val_batch_size"],
            "description":      self.run_description,
        }

    # methods
    def setup_data(self, tensor_size: str, dataset_name: str):
        self._set_seed()
        
        # loading dataset
        dataset = pd.read_csv(os.path.join(self.cfg["csv_path"], dataset_name))
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
        self.target_cols = [
            c for c in dataset.columns
            if c.startswith(self.cfg["target_prefix"])
        ]

        #splitting
        unique_patients = dataset["patient_id"].unique()
        train_ids, temp_ids = train_test_split(unique_patients, test_size=0.30, random_state=self.cfg["random_seed"])
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=self.cfg["random_seed"])
        train_df = self._prepare_split(dataset, train_ids)
        val_df   = self._prepare_split(dataset, val_ids)
        test_df  = self._prepare_split(dataset, test_ids)

        # normalization
        scaler = StandardScaler()
        train_df[self.target_cols] = scaler.fit_transform(train_df[self.target_cols])
        val_df[self.target_cols]   = scaler.transform(val_df[self.target_cols])
        test_df[self.target_cols]  = scaler.transform(test_df[self.target_cols])

        tensor_dir = os.path.join(self.cfg["tensor_dir"], tensor_size)
        bs_train   = self.cfg["train_batch_size"]
        bs_val     = self.cfg["val_batch_size"]
        
        # dataloaders
        self.train_loader = DataLoader(RadiomicDataset(dataset=train_df, tensor_dir=tensor_dir, is_train=True), batch_size=bs_train, shuffle=True, drop_last=True)
        self.val_loader   = DataLoader(RadiomicDataset(dataset=val_df, tensor_dir=tensor_dir, is_train=False), batch_size=bs_val, shuffle=False)
        self.test_loader  = DataLoader(RadiomicDataset(dataset=test_df, tensor_dir=tensor_dir, is_train=False), batch_size=bs_val, shuffle=False)

    def setup_model(self, loss_fn: nn, optimizer: optim, scheduler = None):
        """Instantiate model, loss, optimizer, and scheduler."""
        num_features = len(self.target_cols)

        self.model = self.model_class(num_outputs=num_features, in_channels=self.cfg["in_channels"])

        self.loss_fn   = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Trainable params: {trainable:,} / {total:,}")

    def setup(self, tensor_size: str, dataset_name: str, loss_fn: nn, optimizer: optim, scheduler = None):
        """calls setup_data() then setup_model()."""
        self.setup_data(tensor_size=tensor_size, dataset_name=dataset_name)
        self.setup_model(loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)

    # Training & evaluation 

    def train(self):
        """Run the training loop."""
        self.logger.info("Starting training...")
        self.history = train_model(
            self.model, self.train_loader, self.val_loader,
            self.optimizer, self.loss_fn,
            epochs=self.cfg["epochs"],
            device=self.device,
            scheduler=self.scheduler
        )

    def evaluate(self):
        """Run evaluation on test set and log results to W&B."""
        test_results, fig_eval = evaluate_and_plot(
            self.model, self.test_loader,
            self.target_cols, self.device, self.run_name
        )
        mean_r2 = test_results["R2_Score"].mean()
        self.logger.info(f"Mean Test R²: {mean_r2:.4f}")

        wandb.log({
            "test_r2_report": wandb.Image(fig_eval),
            "mean_test_r2":   mean_r2,
        })
        return test_results

    def save_model(self):
        """Save model weights and log as W&B artifact."""
        os.makedirs("artifacts/saved-models", exist_ok=True)
        model_path = f"artifacts/saved-models/{self.run_name}.pth"
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

        artifact = wandb.Artifact(
            name=self.run_name,
            type="model",
            description=self.run_description,
            metadata=self._build_wandb_config()
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


    def run(self):
        """Full pipeline: train → evaluate → save → finish."""
        wandb.init(
            project=self.cfg["wandb_project"],
            name=self.run_name,
            config=self._build_wandb_config()
        )

        self.train()
        self.evaluate()
        self.save_model()

        plot_loss_curves(self.history)
        wandb.finish()
