import torch, os, logging
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from torchvision import transforms as T
logger = logging.getLogger(__name__)

class RadiomicDataset(Dataset):
    def __init__(self, tensor_dir: str, dataset: pd.DataFrame, target_cols: list, is_train: bool = True):
        self.tensor_dir = tensor_dir
        self.dataset = dataset.reset_index(drop=False) 
        self.target_cols = target_cols
        self.is_train = is_train
        self.transforms = None
        
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        original_idx = self.dataset.iloc[index]['original_idx']
        tensor_path = os.path.join(self.tensor_dir, f"sample_{int(original_idx)}.pt")
        
        x = torch.load(tensor_path)
        # Normalizing HU values
        HU_MIN, HU_MAX = -250.0, 500.0
        x[0] = torch.clamp(x[0], HU_MIN, HU_MAX)
        x[0] = (x[0] - HU_MIN) / (HU_MAX - HU_MIN) # min max normalization for training and for ct scan only
        x[0] = x[0] * x[1]
        y = self.dataset.iloc[index][self.target_cols].values.astype(np.float32)

        return x, torch.tensor(y)