from mirp import extract_features
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass
from mirp.settings.generic import SettingsClass
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
    glcm_spatial_method="2d_average", by_slice=True
)
general_settings = GeneralSettingsClass()
settings = SettingsClass(general_settings=general_settings, feature_extr_settings=feature_settings)

# Radiomics Feature Extraction
def radiomics_features_extraction(image_path: str, mask_path: str, bins_num: int = 16, normalisation_method: str = "none", resample_dim: float | tuple = 1.0):
    MIN_VOXELS_FOR_TEXTURE = 25

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
    zmin, zmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    xmin, xmax = coords[2].min(), coords[2].max()

    margin = 10
    zmin_crop = max(zmin - margin, 0)
    ymin_crop = max(ymin - margin, 0)
    xmin_crop = max(xmin - margin, 0)
    zmax_crop = min(zmax + margin, mask_data.shape[0])
    ymax_crop = min(ymax + margin, mask_data.shape[1])
    xmax_crop = min(xmax + margin, mask_data.shape[2])

    # Save zmin_crop to convert local z_middle back to global coordinates
    zmin_original = zmin_crop

    img_data  = img_data[zmin_crop:zmax_crop, ymin_crop:ymax_crop, xmin_crop:xmax_crop]
    mask_data = mask_data[zmin_crop:zmax_crop, ymin_crop:ymax_crop, xmin_crop:xmax_crop]

    # Recompute coords in local cropped space
    coords = np.where(mask_data > 0)
    zmin_local, zmax_local = coords[0].min(), coords[0].max()
    z_middle_local  = (zmin_local + zmax_local) // 2
    z_middle_global = zmin_original + z_middle_local

    # Extract 2D slice
    img_2d  = img_data[z_middle_local, :, :]
    mask_2d = mask_data[z_middle_local, :, :]

    if np.sum(mask_2d) < MIN_VOXELS_FOR_TEXTURE:
        logger.warning(f"Mask too small for texture features ({np.sum(mask_2d)} voxels): {mask_path}")
        return None, None

    resample_dim = (1.0, 1.0)
    feature_data = extract_features(
        image=img_2d,
        mask=mask_2d,
        new_spacing=resample_dim,
        intensity_normalisation=normalisation_method,
        settings=settings,
        by_slice=True
    )

    del img, mask, img_data, mask_data
    gc.collect()

    return feature_data, z_middle_global

# Dataset Generation Pipeline
patients = os.listdir(PATIENTS_FOLDER_BASE_PATH)
targets = {"pancreas.nii.gz", "pancreas_body.nii.gz", "pancreas_head.nii.gz", 
           "pancreas_tail.nii.gz", "pancreatic_duct.nii.gz", "pancreatic_lesion.nii.gz"}

with ProcessPoolExecutor(max_workers=12) as executor:
    for patient in tqdm(patients, desc="Patients"):
        patient_id = patient.split("_")[1]
        patient_csv = os.path.join(OUTPUT_FOLDER, f"{patient_id}_radiomics_features.csv")
        # Skip if already processed
        if os.path.exists(patient_csv):
            continue
        
        patient_full_path = os.path.join(PATIENTS_FOLDER_BASE_PATH, patient)
        patient_folder = os.listdir(patient_full_path)
        patient_images = [f for f in patient_folder if f.endswith(".nii.gz")]
        if not patient_images:
            logger.warning(f"No image found for patient {patient_id}")
            continue
        patient_image_path = os.path.join(patient_full_path, patient_images[0])

        # Real masks
        segmentation_full_path = os.path.join(SEGMENTATION_FOLDER_BASE_PATH, patient, "segmentations")
        if os.path.exists(segmentation_full_path):
            segmentations = [seg for seg in os.listdir(segmentation_full_path) if seg in targets]
            for organ in segmentations:
                segmentation_mask_path = os.path.join(segmentation_full_path, organ)
                organ_label = organ.split(".")[0]
                try:
                    feature_data_extracted, z_slice = radiomics_features_extraction(patient_image_path, segmentation_mask_path)
                except Exception as e:
                        print(f"Real mask processing failed for patient {patient_id}: {e}")
                        continue
                
                if feature_data_extracted is None:
                    continue
                
                for df in feature_data_extracted:
                    df["z_middle_global"] = z_slice
                    df["patient_id"] = patient_id
                    df["organ"] = organ_label
                    df["is_augmented"] = 0
                    df["ct_image_path"] = patient_image_path
                    df["mask_path"] = segmentation_mask_path
                    df.to_csv(patient_csv, mode="a", header=not os.path.exists(patient_csv), index=False)
                print(f"Successfully processed real mask for patient {patient_id}")
        else:
            logger.warning(f"Real segmentation folder not found for {patient_id}")

        # Augmented masks
        augmented_segmentation_full_path = os.path.join(AUGMENTED_SEGMENTATION_FOLDER_BASE_PATH, patient, "segmentation")
        if os.path.exists(augmented_segmentation_full_path):
            augmented_segmentations = [seg for seg in os.listdir(augmented_segmentation_full_path) if seg in targets]
            for organ in augmented_segmentations:
                augmented_segmentation_mask_path = os.path.join(augmented_segmentation_full_path, organ)
                organ_label = organ.split(".")[0]

                try:
                    feature_data_extracted , z_slice = radiomics_features_extraction(patient_image_path, augmented_segmentation_mask_path)
                except Exception as e:
                        logger.warning(f"Augmented mask processing failed for patient {patient_id}: {e}")
                        continue
                    
                if feature_data_extracted is None:
                    continue
                for df in feature_data_extracted:
                    df["z_middle_global"] = z_slice
                    df["patient_id"] = patient_id
                    df["organ"] = organ_label
                    df["is_augmented"] = 1
                    df["ct_image_path"] = patient_image_path
                    df["mask_path"] = augmented_segmentation_mask_path
                    df.to_csv(patient_csv, mode="a", header=not os.path.exists(patient_csv), index=False)
                print(f"Successfully processed augmented mask for patient {patient_id}")
        else:
            logger.warning(f"Augmented segmentation folder not found for {patient_id}")
         
        gc.collect()


files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.csv"))
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df.to_csv("data/2d_1_slice_texture_radiomics_dataset.csv", index=False)