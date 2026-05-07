import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mirp import extract_features
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass
from mirp.settings.generic import SettingsClass
from dl.helpers import shifted_crop_2d
from dotenv import load_dotenv
from scipy.ndimage import zoom
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import glob, gc, os, logging

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


BASE_PATH = os.getenv("DATASET_BASE_PATH")
PATIENTS_FOLDER_BASE_PATH       = os.path.join(BASE_PATH, "extracted")
SEGMENTATION_FOLDER_BASE_PATH   = os.path.join(BASE_PATH, "labels extracted", "PanTSMini_Label")
AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH = os.getenv("AUGMENTATION_DATASET_BASE_PATH")
OUTPUT_FOLDER                   = os.path.join(BASE_PATH, "generated_dataset")

TENSOR_DIR_10 = "data/processed_tensors/STAT/base_resample"
TENSOR_DIR_12 = "data/processed_tensors/STAT/resample_type_1"
TENSOR_DIR_15 = "data/processed_tensors/STAT/resample_type_2"

FINAL_CSV = "data/raw_dataset/STAT/unified_radiomics_dataset.csv"

for d in [OUTPUT_FOLDER, TENSOR_DIR_12, TENSOR_DIR_15]:
    os.makedirs(d, exist_ok=True)

TARGET_SHAPE     = (256, 256)
MARGIN_RADIOMICS = 20
MARGIN_TENSOR    = 20
MIN_MASK_VOXELS  = 25

RESAMPLINGS = [
    ((1.2, 1.2), TENSOR_DIR_12),
    ((1.5, 1.5), TENSOR_DIR_15),
]

targets = {
    "pancreas.nii.gz",
    "pancreas_body.nii.gz",
    "pancreas_head.nii.gz",
    "pancreas_tail.nii.gz",
    "pancreatic_duct.nii.gz",
    "pancreatic_lesion.nii.gz",
}


feature_settings = FeatureExtractionSettingsClass( base_feature_families="stat", base_discretisation_method="fixed_bin_number", base_discretisation_n_bins=16, by_slice=True )
general_settings = GeneralSettingsClass()
settings = SettingsClass(general_settings=general_settings,feature_extr_settings=feature_settings)


def _get_native_spacing(header) -> tuple:
    """
    Extract in-plane voxel spacing from a NIfTI header.
    """
    zooms = header.get_zooms()
    return (float(zooms[0]), float(zooms[1]))


def _resample_slice( ct_slice: np.ndarray, mask_slice: np.ndarray, native_spacing: tuple, target_spacing: tuple ) -> tuple:
    """
    Resample a 2D CT slice and binary mask from native_spacing to target_spacing.
    Zoom factor = native / target
    """
    zoom_x = native_spacing[0] / target_spacing[0]
    zoom_y = native_spacing[1] / target_spacing[1]

    ct_resampled   = zoom(ct_slice,   (zoom_x, zoom_y), order=3)  # cubic for CT
    mask_resampled = zoom(mask_slice, (zoom_x, zoom_y), order=0)  # nearest for mask
    return ct_resampled, mask_resampled


def _generate_tensor( ct_slice: np.ndarray, mask_slice: np.ndarray, native_spacing: tuple, target_spacing: tuple, tensor_path: Path ) -> bool:
    """
    Resample → crop → resize → save a 2-channel (CT + mask) tensor.
    Returns True on success or if file already exists, False on empty mask.
    """
    if tensor_path.exists():
        return True

    # resample to target voxel spacing
    ct_resampled, mask_resampled = _resample_slice(
        ct_slice, mask_slice, native_spacing, target_spacing
    )

    # verify mask survived resampling
    if np.sum(mask_resampled) == 0:
        logger.warning(f"Mask empty after resampling to {target_spacing}: {tensor_path}")
        return False

    # crop ROI with margin
    ct_crop, mask_crop = shifted_crop_2d(
        ct_slice=ct_resampled,
        mask_slice=mask_resampled,
        margin=MARGIN_TENSOR,
    )

    # resize to fixed network input size
    res_shape  = (TARGET_SHAPE[1], TARGET_SHAPE[0])
    ct_final   = cv2.resize(ct_crop,   res_shape, interpolation=cv2.INTER_LANCZOS4)
    mask_final = cv2.resize(mask_crop, res_shape, interpolation=cv2.INTER_NEAREST)

    # stack channels and save
    x = np.stack([ct_final, mask_final], axis=0)  # (2, 224, 224)
    torch.save(torch.tensor(x, dtype=torch.float32), tensor_path)
    return True


def _extract_base_features( ct_slice: np.ndarray, mask_slice: np.ndarray, normalisation_method: str = "standardisation" ):
    """
    Extract radiomic features at 1.0mm spacing (base resampling).
    These are always the prediction target regardless of input resampling.
    """
    ct_crop, mask_crop = shifted_crop_2d(ct_slice=ct_slice,mask_slice=mask_slice,margin=MARGIN_RADIOMICS)
    return extract_features(
        image=ct_crop,
        mask=mask_crop,
        new_spacing=(1.0, 1.0),
        intensity_normalisation=normalisation_method,
        settings=settings,
        by_slice=True,
    )


# processing
def process_patient(patient: str):
    try:
        patient_id    = patient.split("_")[1]
        patient_csv   = os.path.join(OUTPUT_FOLDER, f"{patient_id}_radiomics_features.csv")

        # skip if this patient was already fully processed
        if os.path.exists(patient_csv):
            return

        # locate CT volume
        patient_full_path = os.path.join(PATIENTS_FOLDER_BASE_PATH, patient)
        patient_images    = [f for f in os.listdir(patient_full_path) if f.endswith(".nii.gz")]
        if not patient_images:
            logger.warning(f"No image found for {patient}")
            return
        patient_image_path = os.path.join(patient_full_path, patient_images[0])

        # load CT volume and extract native spacing — done once per patient
        img            = nib.load(patient_image_path, mmap='r')
        ct_data        = np.asarray(img.dataobj, dtype=np.float32)
        native_spacing = _get_native_spacing(img.header)

        folders = [
            (os.path.join(SEGMENTATION_FOLDER_BASE_PATH, patient, "segmentations"), 0),
            (os.path.join(AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH, patient, "segmentation"), 1),
        ]

        for folder_path, is_aug in folders:
            if not os.path.exists(folder_path):
                continue

            segs = [s for s in os.listdir(folder_path) if s in targets]

            for organ_file in segs:
                organ_name = organ_file.split(".")[0]
                mask_path  = os.path.join(folder_path, organ_file)

                # load mask
                mask       = nib.load(mask_path)
                mask_data  = np.asarray(mask.dataobj, dtype=np.int8)
                mask_data[mask_data > 0] = 1

                if mask_data.sum() == 0:
                    logger.warning(f"Empty mask: {mask_path}")
                    del mask, mask_data
                    continue

                # find middle slice along z axis
                coords          = np.where(mask_data > 0)
                z_middle_global = int((coords[0].min() + coords[0].max()) // 2)

                ct_slice   = ct_data[z_middle_global, :, :]
                mask_slice = mask_data[z_middle_global, :, :].astype(np.uint8)

                if np.sum(mask_slice) < MIN_MASK_VOXELS:
                    logger.warning(
                        f"Mask too small ({int(np.sum(mask_slice))} voxels): {mask_path}"
                    )
                    del mask, mask_data
                    continue

                # deterministic tensor stem — patient_id 
                aug_tag     = "aug" if is_aug else "real"
                tensor_stem = f"pid_{patient_id}_{organ_name}_{aug_tag}"

                # generate tensors for all resampling types
                tensor_paths = {}
                all_tensors_ok = True

                for target_spacing, tensor_dir in RESAMPLINGS:
                    tensor_path = Path(tensor_dir) / f"{tensor_stem}.pt"
                    saved = _generate_tensor( ct_slice=ct_slice, mask_slice=mask_slice, native_spacing=native_spacing, target_spacing=target_spacing, tensor_path=tensor_path)
                    if not saved:
                        logger.warning(
                            f"Tensor generation failed for {tensor_stem} "
                            f"at spacing {target_spacing}"
                        )
                        all_tensors_ok = False
                        break
                    tensor_paths[target_spacing] = str(tensor_path)

                if not all_tensors_ok:
                    del mask, mask_data
                    continue

                # extract 1.0mm radiomic features (always the target)
                feature_data = _extract_base_features(ct_slice, mask_slice)

                if feature_data is None:
                    del mask, mask_data
                    continue

                # write one CSV row per feature DataFrame returned
                for df in feature_data:
                    df["patient_id"]      = patient_id
                    df["z_middle_global"] = z_middle_global
                    df["ct_image_path"]   = patient_image_path
                    df["mask_path"]       = mask_path

                    # one row per resampling type instead of two columns
                    for target_spacing, tensor_dir in RESAMPLINGS:
                        spacing_label = f"{target_spacing[0]}mm"  # "1.2mm" or "1.5mm"
                        row = df.copy()
                        row["resampling"]   = spacing_label
                        row["tensor_path"]  = tensor_paths[target_spacing]

                        row.to_csv(
                            patient_csv,
                            mode="a",
                            header=not os.path.exists(patient_csv),
                            index=False,
                        )

                del mask, mask_data
                gc.collect()

        del img, ct_data
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing {patient}: {e}", exc_info=True)

if __name__ == "__main__":
    patients = os.listdir(PATIENTS_FOLDER_BASE_PATH)

    with ProcessPoolExecutor(max_workers=16) as executor:
        list(tqdm(
            executor.map(process_patient, patients),
            total=len(patients),
            desc="Processing patients",
        ))

    # merge all per-patient CSVs into one final dataset
    files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.csv"))
    if files:
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        df.to_csv(FINAL_CSV, index=False)
        print(f"Done. {len(df)} rows | {df['patient_id'].nunique()} patients → {FINAL_CSV}")
    else:
        print("No output CSVs found. Check logs for errors.")