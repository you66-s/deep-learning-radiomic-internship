import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class MedicalCTDataset(Dataset):

    def __init__(self, image_size=128):
        # default value will be overwritten in train.py with ModelConfig.IMAGE_SIZE
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.BASE_DIR = os.path.normpath(os.path.join(repo_root, "codes", "data", "PanTS", "Extracted"))
        self.image_size = image_size
        self.slices = []

        # parcourir patients
        if not os.path.isdir(self.BASE_DIR):
            return

        for patient in os.listdir(self.BASE_DIR):
            patient_dir = os.path.join(self.BASE_DIR, patient)
            if not os.path.isdir(patient_dir):
                continue
            for scan in os.listdir(patient_dir):
                if scan.endswith(".nii") or scan.endswith(".nii.gz"):
                    full_path = os.path.join(patient_dir, scan)
                    try:
                        image = nib.load(full_path).get_fdata()
                    except Exception:
                        continue

                    # Extract 2D slices
                    for i in range(image.shape[2]):
                        slice = image[:, :, i]
                        if np.mean(slice) > 0:
                            self.slices.append(slice)

        # Debug: number of slices found
        print(f"MedicalCTDataset: found {len(self.slices)} slices in {self.BASE_DIR}")
                            
                            
                            
    def normalize_ct(self, img):
        img = np.clip(img, -1000, 400)   # window CT
        img = (img + 1000) / 1400
        return img.astype(np.float32)
    
    def normalize_mri(self, img):
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-8)
        return img.astype(np.float32)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img = self.slices[idx]
        img = self.normalize_mri(img)
        img = cv2.resize(img, (self.image_size, self.image_size))       # resize
        img = torch.tensor(img).unsqueeze(0)    # tensor shape [1,H,W]
        img = (img - 0.5) / 0.5     # normalize GAN [-1,1]
        return img