from mirp import extract_features
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass
from mirp.settings.generic import SettingsClass

import pandas as pd
import nibabel as nib
import numpy as np
import os
import logging
from tqdm import tqdm

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",)
logger = logging.getLogger(__name__)

# Paths
BASE_PATH = "data"
PATIENTS_FOLDER_BASE_PATH = os.path.join(BASE_PATH, "extracted")
SEGMENTATION_FOLDER_BASE_PATH = os.path.join(BASE_PATH, "labels extracted", "PanTSMini_Label")
AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH = "augmented_data"

# MIRP Settings
feature_settings = FeatureExtractionSettingsClass(base_feature_families="statistical")
general_settings = GeneralSettingsClass()

settings = SettingsClass(general_settings=general_settings,feature_extr_settings=feature_settings,)

# Radiomics Feature Extraction
def radiomics_features_extraction(image_path: str, mask_path: str, bins_num: int = 16, normalisation_method: str = "standardisation", resample_dim: float | tuple = 1.0):

    img = nib.load(image_path)
    img_data = img.get_fdata().astype(np.float32)

    mask = nib.load(mask_path)
    mask_data = mask.get_fdata().astype(np.int32)
    mask_data[mask_data > 0] = 1

    if mask_data.sum() == 0:
        logger.warning(f"Empty mask detected: {mask_path}")
        return None

    feature_data = extract_features(
        image=img_data,
        mask=mask_data,
        new_spacing=resample_dim,
        intensity_normalisation=normalisation_method,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=bins_num,
        settings=settings,
    )

    return feature_data

# Dataset Generation Pipeline
patients = sorted(os.listdir(PATIENTS_FOLDER_BASE_PATH))
targets = {"pancreas.nii.gz", "pancreas_body.nii.gz", "pancreas_head.nii.gz", "pancreas_tail.nii.gz", "pancreatic_duct.nii.gz", "pancreatic_lesion.nii.gz"}
results = []

for patient in tqdm(patients[:5], desc="Patients"):

    patient_id = patient.split("_")[1]
    logger.info(f"Processing patient {patient_id}")

    patient_full_path = os.path.join(PATIENTS_FOLDER_BASE_PATH, patient)

    patient_folder = os.listdir(patient_full_path)

    patient_images = [f for f in patient_folder if f.endswith(".nii.gz")]

    if not patient_images:
        logger.warning(f"No image found for patient {patient_id}")
        continue

    patient_image_path = os.path.join(patient_full_path, patient_images[0])

    segmentation_full_path = os.path.join(
        SEGMENTATION_FOLDER_BASE_PATH,
        patient,
        "segmentations",
    )

    if not os.path.exists(segmentation_full_path):
        logger.warning(f"Real segmentation folder not found for {patient_id}")
        continue

    segmentations = os.listdir(segmentation_full_path)
    segmentations = [seg for seg in segmentations if seg in targets]

    for organ in tqdm(segmentations, desc="Real masks", leave=False):

        segmentation_mask_path = os.path.join(segmentation_full_path, organ)
        organ_label = organ.split(".")[0]

        feature_data_extracted = radiomics_features_extraction(
            image_path=patient_image_path,
            mask_path=segmentation_mask_path,
        )

        if feature_data_extracted is None:
            continue

        for df in feature_data_extracted:
            df["patient_id"] = patient_id
            df["organ"] = organ_label
            df["augmentation"] = 0

        results.extend(feature_data_extracted)

        logger.debug(f"Real mask processed: {organ}")

    augmented_segmentation_full_path = os.path.join(
        AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH,
        patient,
        "segmentation",
    )

    logger.info(f"Looking for augmented folder: {augmented_segmentation_full_path}")

    if not os.path.exists(augmented_segmentation_full_path):
        logger.warning("Augmented segmentation folder not found")
        continue

    augmented_segmentations = os.listdir(augmented_segmentation_full_path)

    logger.debug(f"All augmented files: {augmented_segmentations}")

    augmented_segmentations = [
        seg for seg in augmented_segmentations if seg in targets
    ]

    logger.debug(f"Filtered augmented files: {augmented_segmentations}")

    for organ in tqdm(augmented_segmentations, desc="Augmented masks", leave=False):

        augmented_segmentation_mask_path = os.path.join(
            augmented_segmentation_full_path,
            organ,
        )

        organ_label = organ.split(".")[0]

        feature_data_extracted = radiomics_features_extraction(
            image_path=patient_image_path,
            mask_path=augmented_segmentation_mask_path,
        )

        if feature_data_extracted is None:
            continue

        for df in feature_data_extracted:
            df["patient_id"] = patient_id
            df["organ"] = organ_label
            df["augmentation"] = 1
        results.extend(feature_data_extracted)
        
        logger.debug(f"Augmented mask processed: {organ}")
    logger.info(f"Patient {patient_id} completed")

if len(results) == 0:
    raise ValueError("No radiomics features were extracted.")

feature_data_df = pd.concat(results, ignore_index=True)
cols = (
    ["patient_id", "organ", "augmentation"]
    + [c for c in feature_data_df.columns if c not in ["patient_id", "organ", "augmentation"]]
)

feature_data_df = feature_data_df[cols]
output_path = "data/generated_dataset/generated_radiomics_features.csv"
feature_data_df.to_csv(output_path, index=False)
logger.info(f"Radiomics dataset saved to {output_path}")