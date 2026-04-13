# preprocess.py
import os, logging
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_SHAPE = (128, 128) 
CACHE_DIR = f"data/processed_tensors/{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}_1_slice" 
MARGIN = 10

def pad_volume(volume, target_shape):
    c, d, h, w = volume.shape
    th, tw = target_shape
    if h > th:
        start_h = (h - th) // 2
        volume = volume[:, :, start_h:start_h + th, :]
        h = th
    if w > tw:
        start_w = (w - tw) // 2
        volume = volume[:, :, :, start_w:start_w + tw]
        w = tw
    pad_h, pad_w = max(th - h, 0), max(tw - w, 0)
    padded = np.pad(
        volume,
        ((0,0),(0,0),(pad_h//2, pad_h - pad_h//2),(pad_w//2, pad_w - pad_w//2)),
        mode="constant", constant_values=0
    )
    return padded

def process_one(args):
    idx, row, cache_dir, target_shape = args
    cache_dir = Path(cache_dir)
    out_path  = cache_dir / f"sample_{idx}.pt"

    if out_path.exists():
        return idx, "skipped"

    try:
        ct_data   = nib.load(row["ct_image_path"]).get_fdata(dtype=np.float32)
        mask_data = nib.load(row["mask_path"]).get_fdata(dtype=np.float32)
        mask_data = (mask_data > 0).astype(np.float32)

        if ct_data.shape != mask_data.shape:
            return idx, f"shape mismatch {ct_data.shape} vs {mask_data.shape}"

        z_middle_global = int(row["z_middle_global"])

        if z_middle_global < 0 or z_middle_global >= ct_data.shape[0]:
            return idx, f"z_middle_global={z_middle_global} out of bounds for shape {ct_data.shape}"

        # Compute 3D bounding box from full mask — same crop as radiomics_dataset_generation.py
        coords = np.where(mask_data > 0)
        if coords[0].size == 0:
            return idx, "empty mask"

        margin = 10
        ymin = max(coords[1].min() - margin, 0)
        ymax = min(coords[1].max() + margin, mask_data.shape[1])
        xmin = max(coords[2].min() - margin, 0)
        xmax = min(coords[2].max() + margin, mask_data.shape[2])

        # Extract the exact same 2D crop MIRP received during feature extraction
        ct_slice   = ct_data[z_middle_global, ymin:ymax, xmin:xmax]
        mask_slice = mask_data[z_middle_global, ymin:ymax, xmin:xmax]

        if np.sum(mask_slice) == 0:
            return idx, "empty slice"

        # Add channel dimension → (1, H, W)
        ct_slice   = ct_slice[np.newaxis, :, :]
        mask_slice = mask_slice[np.newaxis, :, :]

        # Stack → (2, H, W)
        x = np.concatenate([ct_slice, mask_slice], axis=0)

        # Pad or crop to target shape using pad_volume
        x = np.expand_dims(x, axis=1)   # (2, 1, H, W)
        x = pad_volume(x, target_shape)  # (2, 1, 128, 128)
        x = x[:, 0, :, :]               # (2, 128, 128)

        torch.save(torch.tensor(x, dtype=torch.float32), out_path)
        return idx, "ok"

    except Exception as e:
        return idx, f"error: {e}"

def preprocess_dataset(csv_path, num_workers=12):
    dataset = pd.read_csv(csv_path)

    # Prépare les arguments pour chaque worker
    args_list = [
        (idx, row, CACHE_DIR, TARGET_SHAPE)
        for idx, row in dataset.iterrows()
    ]

    failed = []
    skipped = 0

    logger.info(f"Lancement avec {num_workers} workers pour {len(dataset)} samples...")

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_one, args_list, chunksize=256),
            total=len(args_list),
            desc="Preprocessing"
        ))

    for idx, status in results:
        if status == "skipped":
            skipped += 1
        elif status != "ok":
            failed.append((idx, status))
            logger.warning(f"Index {idx} échoué : {status}")

    ok_count = len(dataset) - len(failed) - skipped
    logger.info(f"Terminé — ok: {ok_count}, skippés: {skipped}, échoués: {len(failed)}")

    if failed:
        pd.DataFrame(failed, columns=["idx", "reason"]).to_csv(
            "failed_tensors.csv", index=False
        )
        logger.warning(f"Détails des échecs sauvegardés dans {CACHE_DIR}/failed.csv")


if __name__ == "__main__":
    preprocess_dataset("data/processed_dataset/2d_1_slice_texture_radiomics_dataset_clean.csv")