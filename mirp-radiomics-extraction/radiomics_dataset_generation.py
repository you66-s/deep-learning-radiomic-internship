import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mirp import extract_features
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass
from mirp.settings.generic import SettingsClass
from dl.helpers import shifted_crop_2d
from dotenv import load_dotenv
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import glob, gc, os, logging
load_dotenv()

# Logging configuration (only warnings and errors)
logging.basicConfig(
    level=logging.WARNING, 
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_PATH = os.getenv("DATASET_BASE_PATH")
PATIENTS_FOLDER_BASE_PATH = os.path.join(BASE_PATH, "extracted")
SEGMENTATION_FOLDER_BASE_PATH = os.path.join(BASE_PATH, "labels extracted", "PanTSMini_Label")
AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH = os.getenv("AUGMENTATION_DATASET_BASE_PATH")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "generated_dataset")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# MIRP Settings
feature_settings = FeatureExtractionSettingsClass(
    base_feature_families="glcm", base_discretisation_method="fixed_bin_number", base_discretisation_n_bins=16,
    by_slice=True, glcm_spatial_method="2d_average"
)
general_settings = GeneralSettingsClass()
settings = SettingsClass(general_settings=general_settings, feature_extr_settings=feature_settings)
targets = {"pancreas.nii.gz", "pancreas_body.nii.gz", "pancreas_head.nii.gz", "pancreas_tail.nii.gz", "pancreatic_duct.nii.gz", "pancreatic_lesion.nii.gz"}

# Radiomics Feature Extraction
def radiomics_features_extraction(image_path: str, mask_path: str, bins_num: int = 16, normalisation_method: str = "none", resample_dim: float | tuple = 1.0):
    MIN_VOXELS_FOR_TEXTURE = 25
    CROP_SIZE = 220
    MARGIN = 10

    img = nib.load(image_path, mmap='r')
    img_data = np.asarray(img.dataobj, dtype=np.float32)

    mask = nib.load(mask_path)
    mask_data = np.asarray(mask.dataobj, dtype=np.int8)
    mask_data[mask_data > 0] = 1

    if mask_data.sum() == 0:
        logger.warning(f"Empty mask detected: {mask_path}")
        return None, None

    # Crop to bounding box to reduce memory usage
    coords = np.where(mask_data > 0)
    z_middle_global = (coords[0].min() + coords[0].max()) // 2
    
    # middle slice extraction 
    ct_slice = img_data[z_middle_global, :, :]
    mask_slice = mask_data[z_middle_global, :, :]

    if np.sum(mask_slice) < MIN_VOXELS_FOR_TEXTURE:
        logger.warning(f"Mask too small for texture features ({np.sum(mask_slice)} voxels): {mask_path}")
        return None, None
    
    ct_crop, mask_crop = shifted_crop_2d(ct_slice, mask_slice, crop_size=CROP_SIZE + MARGIN)
    
    resample_dim = (1.0, 1.0)
    feature_data = extract_features(
        image=ct_crop,
        mask=mask_crop,
        new_spacing=resample_dim,
        intensity_normalisation=normalisation_method,
        settings=settings,
        by_slice=True
    )

    del img, mask, img_data, mask_data
    gc.collect()

    return feature_data, z_middle_global

def process_patient(patient):
    try:
        patient_id = patient.split("_")[1]
        patient_csv = os.path.join(OUTPUT_FOLDER, f"{patient_id}_radiomics_features.csv")
        if os.path.exists(patient_csv): return
        
        patient_full_path = os.path.join(PATIENTS_FOLDER_BASE_PATH, patient)
        patient_images = [f for f in os.listdir(patient_full_path) if f.endswith(".nii.gz")]
        if not patient_images: return
        patient_image_path = os.path.join(patient_full_path, patient_images[0])

        # Process Real and Augmented Masks
        folders = [
            (os.path.join(SEGMENTATION_FOLDER_BASE_PATH, patient, "segmentations"), 0),
            (os.path.join(AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH, patient, "segmentation"), 1)
        ]

        for folder_path, is_aug in folders:
            if not os.path.exists(folder_path): continue
            
            segs = [s for s in os.listdir(folder_path) if s in targets]
            for organ_file in segs:
                mask_path = os.path.join(folder_path, organ_file)
                feat, z = radiomics_features_extraction(patient_image_path, mask_path)
                
                if feat is not None:
                    for df in feat:
                        df["z_middle_global"] = z
                        df["patient_id"] = patient_id
                        df["organ"] = organ_file.split(".")[0]
                        df["is_augmented"] = is_aug
                        df["ct_image_path"] = patient_image_path
                        df["mask_path"] = mask_path
                        df.to_csv(patient_csv, mode="a", header=not os.path.exists(patient_csv), index=False)
    except Exception as e:
        logger.error(f"Error processing {patient}: {e}")

# Dataset Generation Pipeline
patients = os.listdir(PATIENTS_FOLDER_BASE_PATH)

with ProcessPoolExecutor(max_workers=16) as executor:
    list(tqdm(executor.map(process_patient, patients), total=len(patients), desc="Processing Patients"))


files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.csv"))
if files:
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df.to_csv("data/raw_dataset/GLCM/unique_220_crop_window_glcm_radiomics_dataset.csv", index=False)