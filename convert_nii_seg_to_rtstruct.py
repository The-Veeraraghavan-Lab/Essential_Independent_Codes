#KEY Dependencies: RT_UTILS and NIBABEL
#RT_UTILS: https://github.com/qurit/rt-utils
#NIBABEL: https://nipy.org/nibabel/


import os
import numpy as np
import nibabel as nib
import rt_utils
from datetime import datetime
from rt_utils import RTStructBuilder

# Define the patient ID (scrubbed SHAkey)
patient_id = 'xyz1234'

# Define file paths for input NIfTI segmentation and DICOM series
nifti_segmentation_path = f"segmentations_nii/{patient_id}_predseg.nii.gz"
dicom_reference_path = f"DCMS/{patient_id}"

# Load the DICOM image series and initialize a new RTSTRUCT file
rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_reference_path)

# Load the NIfTI segmentation file
nifti_img = nib.load(nifti_segmentation_path)
segmentation_data = nifti_img.get_fdata()  # Convert to NumPy array

# Ensure integer labels by rounding and converting to uint8
segmentation_data = np.round(segmentation_data).astype(np.uint8)

# Extract unique labels in the segmentation (excluding background, assumed as 0)
unique_labels = np.unique(segmentation_data)
unique_labels = unique_labels[unique_labels != 0]  # Ignore background

# Define class names for each segmentation label (customize as needed)
label_names = {
    1: "Liver",
    2: "Kidney",
    3: "Trachea",
    4: "SpinalCanal",
    5: "Lungs",
    6: "Heart",
    7: "Esophagus",
    8: "Bronchus",
    9: "Aorta",
    10: "PulmonaryArtery",
    11: "PulmonaryVein"
}

# Iterate through each unique label and add it as an ROI to RTSTRUCT
for label in unique_labels:
    mask = segmentation_data == label  # Create a binary mask for the current label
    if np.any(mask):  # Ensure the mask contains data
        roi_name = label_names.get(label, f"Class_{label}")  # Default to "Class_X" if not defined
        rtstruct.add_roi(mask=mask, name=roi_name)

# Save the generated RTSTRUCT DICOM file
rtstruct.save(f"{patient_id}.dcm")

print(f"RTSTRUCT file saved as {patient_id}.dcm")
