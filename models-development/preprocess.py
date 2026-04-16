# preprocess.py
import cv2, logging
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_SHAPE = (224, 224) 
CACHE_DIR = f"data/processed_tensors/STAT/{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}_scaled" 
MARGIN = 10

def process_one(args):
    idx, row, cache_dir, target_shape = args
    out_path = Path(cache_dir) / f"sample_{idx}.pt"

    if out_path.exists():
        return idx, "skipped"

    try:
        # Loading Data
        ct_data   = nib.load(row["ct_image_path"]).get_fdata(dtype=np.float32)
        mask_data = nib.load(row["mask_path"]).get_fdata(dtype=np.float32)
        mask_data = (mask_data > 0).astype(np.float32)

        z_middle_global = int(row["z_middle_global"])

        # Define Bounding Box
        coords = np.where(mask_data > 0)
        if coords[0].size == 0:
            return idx, "empty mask"

        ymin = max(coords[1].min() - MARGIN, 0)
        ymax = min(coords[1].max() + MARGIN, mask_data.shape[1])
        xmin = max(coords[2].min() - MARGIN, 0)
        xmax = min(coords[2].max() + MARGIN, mask_data.shape[2])

        # 3. Extract ROI Slice
        ct_slice   = ct_data[z_middle_global, ymin:ymax, xmin:xmax]
        mask_slice = mask_data[z_middle_global, ymin:ymax, xmin:xmax]

        if np.sum(mask_slice) == 0:
            return idx, "empty slice at z_middle"

        # Scaling
        res_shape = (target_shape[1], target_shape[0]) 

        # CT uses Linear interpolation for smoothness
        ct_resized = cv2.resize(ct_slice, res_shape, interpolation=cv2.INTER_LANCZOS4)
        
        # Mask MUST use Nearest Neighbor to stay binary (0.0 or 1.0)
        mask_resized = cv2.resize(mask_slice, res_shape, interpolation=cv2.INTER_NEAREST)

        # Stack and Save
        # Add channel dimension -> (2, 128, 128)
        x = np.stack([ct_resized, mask_resized], axis=0)
        
        torch.save(torch.tensor(x, dtype=torch.float32), out_path)
        return idx, "ok"

    except Exception as e:
        return idx, f"error: {e}"

def preprocess_dataset(csv_path, num_workers=12):
    dataset = pd.read_csv(csv_path)
    args_list = [(idx, row, CACHE_DIR, TARGET_SHAPE) for idx, row in dataset.iterrows()]

    logger.info(f"Scaling {len(dataset)} samples to {TARGET_SHAPE}...")

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one, args_list, chunksize=256), total=len(args_list)))

    # Process results summary
    failed = [res for res in results if res[1] not in ["ok", "skipped"]]
    logger.info(f"Done. Failures: {len(failed)}")
    if failed:
        pd.DataFrame(failed, columns=["idx", "reason"]).to_csv("failed_scaling.csv", index=False)

if __name__ == "__main__":
    preprocess_dataset("data/processed_dataset/STAT/2d_1_slice_statistical_radiomics_dataset_cleaned.csv")