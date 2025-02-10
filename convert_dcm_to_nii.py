import SimpleITK as sitk
import os

def find_dcm_folders(root_dir):
    """
    Recursively finds all unique folders containing DICOM (.dcm) files.

    :param root_dir: The path to the dataset directory.
    :return: A set of folder paths containing .dcm files.
    """
    dcm_folders = set()

    for dirpath, _, filenames in os.walk(root_dir):
        if any(file.endswith(".dcm") for file in filenames):  # Check if .dcm files exist in the folder
            dcm_folders.add(dirpath)

    return sorted(dcm_folders)  # Sorting for better readability


def dicom_to_nifti(dicom_dir, output_nifti_path):
    """
    Converts a DICOM series to a NIfTI file.

    :param dicom_dir: Path to the directory containing DICOM series.
    :param output_nifti_path: Path to save the output NIfTI file.
    """
    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_series)
    
    # Load image
    image = reader.Execute()
    
    # Save as NIfTI
    sitk.WriteImage(image, output_nifti_path)
    print(f"Saved NIfTI file: {output_nifti_path}")

# # Example usage
# dicom_dir = "path/to/dicom/folder"  # Change this to your DICOM folder path
# output_nifti_path = "output.nii.gz"  # Change this to desired output path

# dicom_to_nifti(dicom_dir, output_nifti_path)

dicom_dir_paths = find_dcm_folders('MIDRC-RICORD')

for idx in range(len(dicom_dir_paths)):

    dicom_path = dicom_dir_paths[idx]
    filename = os.path.join('COVID-19','{}.nii.gz'.format(dicom_path.split("\\")[3]))
    
    dicom_to_nifti(dicom_path, filename)