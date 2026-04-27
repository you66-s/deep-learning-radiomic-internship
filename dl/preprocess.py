# preprocess.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2, logging
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from dl.helpers import shifted_crop_2d

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_SHAPE = (224, 224) 
CACHE_DIR = f"data/processed_tensors/GLCM/{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}_220_crop" 
CROP_SIZE = 220
MARGIN = 10

def process_one(args):
    idx, row, cache_dir, target_shape = args
    out_path = Path(cache_dir) / f"sample_{idx}.pt"

    if out_path.exists():
        return idx, "skipped"

    try:
        # Loading Data
        img = nib.load(row["ct_image_path"])
        spacing_z, spacing_y, spacing_x = img.header.get_zooms()
        ct_data = np.asarray(img.dataobj, dtype=np.float32)
        mask_data = nib.load(row["mask_path"]).get_fdata(dtype=np.float32)
        mask_data = (mask_data > 0).astype(np.float32)

        z_middle_global = int(row["z_middle_global"])

        # 3. Extract ROI Slice
        ct_slice   = ct_data[z_middle_global, :, :]
        mask_slice = mask_data[z_middle_global, :, :]

        if np.sum(mask_slice) == 0:
            return idx, f"empty slice at z_middle: {z_middle_global}"
        
        ct_crop, mask_crop = shifted_crop_2d(ct_slice=ct_slice, mask_slice=mask_slice, crop_size=CROP_SIZE + MARGIN)
        
        # Resampling to 1.0mm
        #resample_w = int(round(ct_crop.shape[1] * spacing_x))
        #resample_h = int(round(ct_crop.shape[0] * spacing_y))
        #ct_resampled = cv2.resize(ct_crop, (resample_w, resample_h), interpolation=cv2.INTER_LANCZOS4)
        #mask_resampled = cv2.resize(mask_crop, (resample_w, resample_h), interpolation=cv2.INTER_NEAREST)
        
        # Scaling
        res_shape = (target_shape[1], target_shape[0]) 
        ct_final = cv2.resize(ct_crop, res_shape, interpolation=cv2.INTER_LANCZOS4)
        # Nearest Neighbor to stay binary (0.0 or 1.0)
        mask_final = cv2.resize(mask_crop, res_shape, interpolation=cv2.INTER_NEAREST)

        # Stack and Save
        # Add channel dimension -> (2, 224, 224)
        x = np.stack([ct_final, mask_final], axis=0)
        
        torch.save(torch.tensor(x, dtype=torch.float32), out_path)
        return idx, "ok"

    except Exception as e:
        return idx, f"error: {e}"

def preprocess_dataset(csv_path, num_workers=16):
    dataset = pd.read_csv(csv_path)
    args_list = [(idx, row, CACHE_DIR, TARGET_SHAPE) for idx, row in dataset.iterrows()]

    logger.info(f"Scaling {len(dataset)} samples to {TARGET_SHAPE}...")

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one, args_list, chunksize=64), total=len(args_list)))

    # Process results summary
    failed = [res for res in results if res[1] not in ["ok", "skipped"]]
    logger.info(f"Done. Failures: {len(failed)}")
    if failed:
        pd.DataFrame(failed, columns=["idx", "reason"]).to_csv("failed_scaling.csv", index=False)

if __name__ == "__main__":
    preprocess_dataset("data/processed_dataset/GLCM/unique_220_crop_window_glcm_radiomics_dataset_cleaned.csv")