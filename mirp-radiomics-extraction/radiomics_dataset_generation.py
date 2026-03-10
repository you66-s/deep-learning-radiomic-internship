from mirp import extract_features
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass
from mirp.settings.generic import SettingsClass
import pandas as pd
import nibabel as nib
import numpy as np
import os

BASE_PATH = r"data"
PATIENTS_FOLDER_BASE_PATH = os.path.join(BASE_PATH, "extracted")
SEGMENTATION_FOLDER_BASE_PATH = os.path.join(BASE_PATH, "labels extracted", "PanTSMini_Label")
AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH = os.path.join("augmented_data")


feature_settings = FeatureExtractionSettingsClass(
    base_feature_families="statistical"
)

general_settings = GeneralSettingsClass()
settings = SettingsClass(
    general_settings=general_settings,
    feature_extr_settings=feature_settings,
)

def radiomics_features_extraction(image_path: str, mask_path: str,  bins_num: int = 16, normalisation_method: str = "standardisation", resample_dim: float | tuple = 1.0) -> list:
    img = nib.load(image_path)
    img_data = img.get_fdata()
    # Convertir en int32
    img_data_int = img_data.astype(np.float32)
    # Charger mask binaire
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata().astype(np.int32)
    mask_data[mask_data > 0] = 1

    feature_data = extract_features(
        image=img_data_int,
        mask=mask_data,
        new_spacing=resample_dim,
        intensity_normalisation=normalisation_method,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=bins_num,
        settings=settings
    )
    return feature_data


# Pipeline of batch statistical radiomic features extraction
patients = os.listdir(path=PATIENTS_FOLDER_BASE_PATH)
targets = {'pancreas.nii.gz', 'pancreas_body.nii.gz', 'pancreas_head.nii.gz', 'pancreas_tail.nii.gz', 'pancreatic_duct.nii.gz', 'pancreatic_lesion.nii.gz'} 
results = []

for patient in patients[:5]:
    patient_id_parts = patient.split("_")
    patient_id = patient_id_parts[1] if len(patient_id_parts) > 1 else patient
    patient_full_path = os.path.join(PATIENTS_FOLDER_BASE_PATH, patient)
    patient_folder = os.listdir(patient_full_path)
    patient_images = [f for f in patient_folder if f.endswith(".nii.gz")]
    if not patient_images:
        continue
    patient_image_path = os.path.join(patient_full_path, patient_images[0])
    
    segmentation_full_path = os.path.join(SEGMENTATION_FOLDER_BASE_PATH, patient, "segmentations")     # full path of folder of segmentations for each patient
    if not os.path.exists(segmentation_full_path):
        continue
    segmentations = os.listdir(segmentation_full_path)
    segmentations = [seg for seg in segmentations if seg in targets]
    
    for organ in segmentations:
        segmentation_mask_path = os.path.join(segmentation_full_path, organ)    # full mask path
        organ_label = organ.split(".")[0]
        feature_data_extracted = radiomics_features_extraction(image_path=patient_image_path, mask_path=segmentation_mask_path)
        for df in feature_data_extracted:
            df["patient_id"] = patient_id
            df["organ"] = organ_label
            df["augmentation"] = 0
        results.extend(feature_data_extracted)
    
    augmented_segmentation_full_path = os.path.join(AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH, patient)     # full path of folder of segmentations for each patient
    if not os.path.exists(augmented_segmentation_full_path):
        continue
    augmented_segmentations = os.listdir(augmented_segmentation_full_path)
    augmented_segmentations = [seg for seg in augmented_segmentations if seg in targets]
    
    for organ in augmented_segmentations:
        augmented_segmentation_mask_path = os.path.join(augmented_segmentation_full_path, organ)    # full mask path
        organ_label = organ.split(".")[0]
        feature_data_extracted = radiomics_features_extraction(image_path=patient_image_path, mask_path=augmented_segmentation_mask_path)
        for df in feature_data_extracted:
            df["patient_id"] = patient_id
            df["organ"] = organ_label
            df["augmentation"] = 1
        results.extend(feature_data_extracted)

if len(results) == 0:
    raise ValueError("No radiomics features were extracted.")

feature_data_df = pd.concat(results, ignore_index=True)
cols = ["patient_id", "organ", "augmentation"] + [c for c in feature_data_df.columns if c not in ["patient_id", "organ", "augmentation"]]
feature_data_df = feature_data_df[cols]
feature_data_df.to_csv("generated_radiomics_features.csv", index=False)