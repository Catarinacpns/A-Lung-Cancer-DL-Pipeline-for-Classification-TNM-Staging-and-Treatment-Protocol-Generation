# Imports
import os

# External libraries
import pydicom
import xml.etree.ElementTree as ET

# Custom project imports
from src.preprocessing.uid_mapping import getUID_path
from src.preprocessing.dicom_io import read_dicom_image
from src.preprocessing.xml_parsing import extract_bounding_boxes


def create_dataset(image_dir, annotation_dir):
    """
    Creates a dataset by matching DICOM images with their corresponding XML annotations.

    Args:
        image_dir (str): Directory containing DICOM images organized by patient folders.
        annotation_dir (str): Directory containing XML annotation files.

    Returns:
        list: A list of dictionaries, each representing an image and its metadata.
    """
    dataset = []
    uid_dict = getUID_path(image_dir)  # Retrieve UID and metadata for DICOM files
    
    for uid, (image_path, patient_id, description, subdescription, filename) in uid_dict.items():
        annotation_found = False  # Flag to track if the annotation was found
        for root, dirs, files in sorted(os.walk(annotation_dir)):
            for file in sorted(files):
                if file == f"{uid}.xml":  # Match annotation file with UID
                    annotation_path = os.path.join(root, file)
                    annotation_found = True

                    # Load image and bounding boxes
                    try:
                        image = read_dicom_image(image_path)
                        bboxes = extract_bounding_boxes(annotation_path)
                    except Exception as e:
                        print(f"Error processing UID {uid}: {e}")
                        continue

                    # Append to dataset
                    dataset.append({
                        "Uid": uid,
                        "Image": image,
                        "Image Path": image_path,
                        "BoundingBoxes": bboxes,
                        "Patient ID": patient_id,
                        "Description": description,
                        "Subdescription": subdescription,
                        "Filename": filename,
                        "Annotation Path": annotation_path,
                    })
                    break  # Stop searching for this UID once matched

    return dataset



def get_images_by_patient_id(dataset, patient_id):
    """
    Retrieves images and relevant metadata for a specific patient ID.

    Args:
        dataset (list): A list of dictionaries representing images and metadata.
        patient_id (str): The patient ID to filter.

    Returns:
        list: A list of tuples (Image, BoundingBoxes, Description, Subdescription, Filename).
    """
    filtered_data = []

    for entry in dataset:
        try:
            if entry.get("Patient ID") == patient_id:  # Use .get() to avoid KeyError
                filtered_data.append((
                    entry["Image"],
                    entry["BoundingBoxes"],
                    entry["Description"],
                    entry["Subdescription"],
                    entry["Filename"]
                ))
        except KeyError as e:
            print(f"Skipping entry due to missing key {e}: {entry}")

    return filtered_data