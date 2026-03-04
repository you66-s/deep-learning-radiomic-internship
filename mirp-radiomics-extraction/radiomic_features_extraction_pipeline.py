from mirp import extract_features
import pandas as pd
import nibabel as nib
import numpy as np
import os

# Constants 
BASE_PATH = r"mirp-radiomics-extraction\data\PanTS"
PATIENTS_FOLDER_BASE_PATH = os.path.join(BASE_PATH, "Extracted")    # full path that contains the patients
SEGMENTATION_FOLDER_BASE_PATH = os.path.join(BASE_PATH, "Labels Extracted") # full path that contains the segmentations for each patient

# Functions
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
        intensity_normalisation="standardisation",
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=16
    )
    return feature_data

# Pipeline of batch radiomic features extraction
patients = os.listdir(path=PATIENTS_FOLDER_BASE_PATH)   # list of patients
targets = {'pancreas.nii.gz', 'liver.nii.gz'}
results = []

for patient in patients[:25]:
    patient_id = patient.split("_")[1]
    patient_full_path = os.path.join(PATIENTS_FOLDER_BASE_PATH, patient)  # single patient full path folder
    patient_folder = os.listdir(patient_full_path)  # single patient folder data
    patient_images = [f for f in patient_folder if f.endswith(".nii.gz")]
    if not patient_images:
        continue
    patient_image_path = os.path.join(patient_full_path, patient_images[0])    # full image path
    
    segmentation_full_path = os.path.join(SEGMENTATION_FOLDER_BASE_PATH, patient, "segmentations")     # full path of folder of segmentations for each patient
    segmentations = os.listdir(segmentation_full_path)
    segmentations = [seg for seg in segmentations if seg in targets]
    
    for organ in segmentations:
        segmentation_mask_path = os.path.join(segmentation_full_path, organ)    # full mask path
        organ_label = organ.split(".")[0]
        print(f"image path: {patient_image_path}")
        print(f"mask path: {segmentation_mask_path}")
        feature_data_extracted = radiomics_features_extraction(image_path=patient_image_path, mask_path=segmentation_mask_path)
        for df in feature_data_extracted:
            df["patient_id"] = patient_id
            df["organ"] = organ_label
        results += feature_data_extracted
        

feature_data_df = pd.concat(results, ignore_index=True)
cols = ["patient_id", "organ"] + [c for c in feature_data_df.columns if c not in ["patient_id", "organ"]]
feature_data_df = feature_data_df[cols]
feature_data_df.to_csv("liver_4_radiomics_features.csv", index=False)