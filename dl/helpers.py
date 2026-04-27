import logging, os
import numpy as np

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

def shifted_crop_2d(ct_slice, mask_slice, crop_size):
    """
    Center crop on ROI, but shift the window if it goes out of boundaries. And no padding
    """

    yx = np.where(mask_slice > 0)

    # safety check
    if len(yx[0]) == 0:
        return None, None, {"error": "empty mask"}

    # ROI center (bounding box center)
    cy = int((yx[0].min() + yx[0].max()) / 2)
    cx = int((yx[1].min() + yx[1].max()) / 2)
    
    half = crop_size // 2

    # initial window
    y1 = cy - half
    y2 = cy + half
    x1 = cx - half
    x2 = cx + half

    img_h, img_w = ct_slice.shape
    if y1 < 0:
        y2 += -y1
        y1 = 0
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    if x1 < 0:
        x2 += -x1
        x1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w

    # final safety (important for extreme cases)
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(img_h, y1 + crop_size)
    x2 = min(img_w, x1 + crop_size)

    ct_crop   = ct_slice[y1:y2, x1:x2]
    mask_crop = mask_slice[y1:y2, x1:x2]

    # ROI size
    roi_h = int(yx[0].max() - yx[0].min())
    roi_w = int(yx[1].max() - yx[1].min())

    meta = {
        "cy": cy, "cx": cx,
        "roi_h": roi_h, "roi_w": roi_w,
        "roi_max": max(roi_h, roi_w),
        "clipped": roi_h > crop_size or roi_w > crop_size,
        "shifted": (
            cy - half < 0 or cy + half > img_h or
            cx - half < 0 or cx + half > img_w
        )
    }

    return ct_crop, mask_crop, meta

def shifted_crop_2d(ct_slice, mask_slice, crop_size):

    yx = np.where(mask_slice > 0)
    if len(yx[0]) == 0:
        return None, None, {"error": "empty mask"}

    cy = int((yx[0].min() + yx[0].max()) / 2)
    cx = int((yx[1].min() + yx[1].max()) / 2)
    
    half = crop_size // 2
    img_h, img_w = ct_slice.shape
    
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    if y1 < 0:
        y2 += -y1
        y1 = 0
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    if x1 < 0:
        x2 += -x1
        x1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w

    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = min(img_h, y1 + crop_size), min(img_w, x1 + crop_size)

    return ct_slice[y1:y2, x1:x2], mask_slice[y1:y2, x1:x2]