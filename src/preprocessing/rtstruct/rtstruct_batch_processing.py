import os

# RTSTRUCT loading and CT loading
from .rtstruct_io import load_rtstruct, load_ct_images

# Contour → bounding box extraction
from .rtstruct_parsing import extract_tumor_bboxes

# YOLO output generation
from .rtstruct_to_yolo import save_yolo_format, save_images_as_jpeg


def process_all_patients(data_dir, output_image_dir, output_label_dir):
    """Iterates through all patient directories, extracts and saves YOLO annotations and images."""
    for patient in sorted(os.listdir(data_dir)):
        patient_path = os.path.join(data_dir, patient)
        if os.path.isdir(patient_path):
            study_folders = [f for f in sorted(os.listdir(patient_path)) if os.path.isdir(os.path.join(patient_path, f))]
            for study_folder_name in study_folders:
                study_folder = os.path.join(patient_path, study_folder_name)
                
                ct_dir = find_ct_folder(study_folder)  # Dynamically find CT folder
                rtstruct_path = find_rtstruct_folder(study_folder)  # Dynamically find RTSTRUCT file
                
                if not ct_dir or not rtstruct_path:
                    print(f"Skipping {patient} - Missing CT or RTSTRUCT file in {study_folder}")
                    continue
                
                print(f"Processing {patient} - {study_folder_name}...")
                
                try:
                    rtstruct, roi_map = load_rtstruct(rtstruct_path)
                except ValueError as e:
                    print(f"Skipping {rtstruct_path}: {e}")
                    continue
                
                ct_slices = load_ct_images(ct_dir)
                bboxes = extract_tumor_bboxes(rtstruct, roi_map, ct_slices)
                
                if bboxes:
                    save_yolo_format(patient, study_folder_name, bboxes, output_label_dir)
                    save_images_as_jpeg(patient, study_folder_name, ct_slices, bboxes, output_image_dir)