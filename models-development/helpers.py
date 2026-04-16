import logging, os

logger = logging.getLogger(__name__)

def check_tensor_integrity(df, tensor_dir, split_name):
    """Verifies that every index in the dataframe has a corresponding tensor file."""
    missing_indices = []
    for idx in df["original_idx"]:
        file_path = os.path.join(tensor_dir, f"sample_{idx}.pt")
        if not os.path.exists(file_path):
            missing_indices.append(idx)
    
    if missing_indices:
        logger.error(f"{split_name} MISMATCH: {len(missing_indices)} tensors missing from {tensor_dir}")
        logger.error(f"First 5 missing indices: {missing_indices[:5]}")
        # sys.exit(1) 
    else:
        logger.info(f"{split_name} Integrity Check: All {len(df)} tensors found.")
    return missing_indices