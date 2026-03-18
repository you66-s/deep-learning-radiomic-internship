import torch, os, logging
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from torchvision import transforms as T
logger = logging.getLogger(__name__)

class RadiomicDataset(Dataset):
    def __init__(self, tensor_dir: str, csv_dataset: pd.DataFrame, is_train: bool = True):
        self.tensor_dir = tensor_dir
        self.dataset = csv_dataset.reset_index(drop=False) 
        self.target_cols = [col for col in self.dataset.columns if col.startswith("stat_")]
        self.is_train = is_train
        self.transforms = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomRotation(15),
        ])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original_idx = self.dataset.iloc[index]['index']
        tensor_path = os.path.join(self.tensor_dir, f"sample_{int(original_idx)}.pt")
        x = torch.load(tensor_path)
        if self.is_train:
            x = self.transforms(x)
        y = self.dataset.iloc[index][self.target_cols].values.astype(np.float32)

        return x, torch.tensor(y)