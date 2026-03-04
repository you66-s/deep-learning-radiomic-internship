import mlflow
import mlflow.pytorch
import os

import torchvision.utils as vutils
import matplotlib.pyplot as plt

class MLflowTracker:
    def __init__(self, experiment_name, run_name: str | None):
        self.run_name = run_name
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)
        print(f"MLflow Run Started: {self.run.info.run_id}")

    def set_run_name(self, run_name: str):
        self.run_name = run_name
    
    def log_config(self, config_obj):
        params = {attr: getattr(config_obj, attr) for attr in dir(config_obj) 
                  if not attr.startswith("__") and not callable(getattr(config_obj, attr))}
        mlflow.log_params(params)

    def log_metrics(self, metrics, step):
        mlflow.log_metrics(metrics, step=step)

    def log_image_grid(self, tensor_batch, title, step):
        grid = vutils.make_grid(tensor_batch, padding=2, normalize=True)
        # Convert to numpy for matplotlib
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(title)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        
        temp_path = f"temp_grid_{step}.png"
        plt.savefig(temp_path, bbox_inches="tight")
        plt.close()
        
        mlflow.log_artifact(temp_path, artifact_path="generated_images")
        os.remove(temp_path)

    def log_models(self, netG, netD):
        """Logs the PyTorch models to the MLflow Model Registry."""
        mlflow.pytorch.log_model(netG, "generator_model")
        mlflow.pytorch.log_model(netD, "discriminator_model")

    def end_run(self):
        mlflow.end_run()