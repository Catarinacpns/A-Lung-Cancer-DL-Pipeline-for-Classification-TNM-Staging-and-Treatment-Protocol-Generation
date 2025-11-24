import pickle
import pydicom as dicomio
import os
from collections import Counter


def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def loadFileInformation(filename):
    ''' Extract and return metadata from a DICOM file, such as the SOPInstanceUID.'''
    
    information = {}
    ds = dicomio.read_file(filename, force=True)
    information['dicom_num'] = ds.SOPInstanceUID
    # information['PatientID'] = ds.PatientID
    # information['PatientName'] = ds.PatientName
    # information['PatientBirthDate'] = ds.PatientBirthDate
    # information['PatientSex'] = ds.PatientSex
    # information['StudyID'] = ds.StudyID
    # information['StudyDate'] = ds.StudyDate
    # information['StudyTime'] = ds.StudyTime
    # information['InstitutionName'] = ds.InstitutionName
    # information['Manufacturer'] = ds.Manufacturer
    # information['NumberOfFrames'] = ds.NumberOfFrames
    return information

def rename_files_in_folder(folder_path, patient_to_histology, histology_prefix):
    """Renames files in the folder by adding the correct histology letter prefix."""
    for filename in os.listdir(folder_path):
        # Extract patient ID from the file name (first part before "_")
        patient_id = filename.split("_")[0]

        # Get corresponding histology and prefix
        histology = patient_to_histology.get(patient_id, None)
        if histology in histology_prefix:
            prefix = histology_prefix[histology]
            new_filename = f"{prefix}_{filename}"

            # Rename file
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")
            

def count_files_by_prefix(folder_path, prefixes):
    """Counts how many files in a folder start with each prefix."""
    counts = Counter()
    
    for filename in os.listdir(folder_path):
        for prefix in prefixes:
            if filename.startswith(prefix):
                counts[prefix] += 1

    return counts


def update_label_files(label_dir, histology_class_map):
    """Updates YOLO label files by replacing the first number with the correct class."""
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            histology_prefix = filename[0]  # Extract the first letter
            
            if histology_prefix in histology_class_map:
                class_number = histology_class_map[histology_prefix]
                
                label_path = os.path.join(label_dir, filename)
                
                # Read and modify the label file
                with open(label_path, "r") as f:
                    lines = f.readlines()

                # Replace first number in each line
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # Ensure correct YOLO format
                        parts[0] = class_number  # Replace class ID
                        updated_lines.append(" ".join(parts))

                # Write updated lines back to the file
                with open(label_path, "w") as f:
                    f.write("\n".join(updated_lines) + "\n")

                print(f"Updated: {filename}")