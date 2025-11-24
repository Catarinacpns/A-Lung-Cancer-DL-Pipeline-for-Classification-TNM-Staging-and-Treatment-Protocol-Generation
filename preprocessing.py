def extract_number(s):
    # Extracts numbers from the string, ignoring leading characters
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else 0


def count_unique_subject_ids(df, letter):
    filtered = metadata_sampled[metadata_sampled['Subject ID'].str.startswith(letter)]
    unique_count = filtered['Subject ID'].str[1:].nunique()
    return unique_count


import os
import xml.etree.ElementTree as ET
import pydicom
import cv2
import numpy as np
import pandas as pd
import pickle
import albumentations as A
import time
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.model_selection import train_test_split
from collections import defaultdict

from VisualizationTools.get_data_from_XML import XML_preprocessor, get_category
from VisualizationTools.get_gt import get_gt
from VisualizationTools.getUID import getUID_path
from VisualizationTools.utils import loadFileInformation
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, Affine,
    GaussianBlur, CLAHE, RandomGamma, GaussNoise
)

import shutil
from collections import defaultdict
from random import shuffle

# OS and File Management
import os
import shutil
import random
from glob import glob
import pickle
import xml.etree.ElementTree as ET
import re

# Scientific Computing and Data Processing
import numpy as np
import pandas as pd

# Image Processing and Augmentation
import cv2
import pydicom
import albumentations as A
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, Affine,
    GaussianBlur, CLAHE, RandomGamma, GaussNoise
)

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Machine Learning Utilities
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Custom Utilities
from VisualizationTools.get_data_from_XML import XML_preprocessor, get_category
from VisualizationTools.get_gt import get_gt
from VisualizationTools.getUID import getUID_path
from VisualizationTools.utils import loadFileInformation
from functions_DataPreprocessing import *



# Function to sample data with focus on maximizing number of images while keeping all categories
def relaxed_stratified_sample(df, max_images, max_memory, stage_columns, seed=1):
    # Set a seed for reproducibility
    np.random.seed(seed)

    # Drop rows with missing stage values (if any)
    df = df.dropna(subset=stage_columns)

    # Group by stage columns to ensure each category is present at least once
    grouped = df.groupby(stage_columns)

    # Sample each group to ensure at least one sample per group, if possible
    sampled_df = pd.DataFrame()
    total_images_sampled = 0
    total_memory_used = 0
    
    # First ensure all groups are represented by at least one image
    for group_name, group in grouped:
        if not group.empty:
            sample = group.sample(n=min(1, len(group)), random_state=seed)  # Ensure at least one row per group if possible
            sampled_df = pd.concat([sampled_df, sample])
            total_images_sampled += sample['Number of Images'].sum()
            total_memory_used += sample['File Size'].sum()

    # Sample the rest of the data prioritizing the image and memory limits for balancing
    remaining_images = max_images - total_images_sampled
    remaining_memory = max_memory - total_memory_used

    if remaining_images > 0 and remaining_memory > 0:
        # Sort by file size for efficient memory usage
        remaining_df = df[~df.index.isin(sampled_df.index)].sort_values(by='File Size')
        remaining_df = remaining_df[(remaining_df['File Size'].cumsum() <= remaining_memory) & 
                                    (remaining_df['Number of Images'].cumsum() <= remaining_images)]
        
        sampled_df = pd.concat([sampled_df, remaining_df])
        total_images_sampled += remaining_df['Number of Images'].sum()
        total_memory_used += remaining_df['File Size'].sum()

    return sampled_df, total_memory_used

'''_______________________________________________________________________________________________________________'''

# Function to add more patients if space remains
def add_more_patients(df, sampled_df, remaining_images, remaining_memory, seed=19):
    if remaining_images > 0 and remaining_memory > 0:
        remaining_df = df[~df.index.isin(sampled_df.index)].sort_values(by='File Size')
        additional_sample = remaining_df[
            (remaining_df['File Size'].cumsum() <= remaining_memory) & 
            (remaining_df['Number of Images'].cumsum() <= remaining_images)
        ]
        sampled_df = pd.concat([sampled_df, additional_sample])
    return sampled_df

'''_______________________________________________________________________________________________________________'''

# Modify the get_target_sample function to keep all patients in B and E, and sample A and G
def get_target_sample(dfs, max_images_per_df, remaining_memory, stage_columns, seed=1):
    sampled_dfs = {'E': dfs['E'], 'B': dfs['B']}  # Include all data from E and B
    remaining_memory -= dfs['E']['File Size'].sum()  # Update remaining memory after including E dataset
    remaining_memory -= dfs['B']['File Size'].sum()  # Update remaining memory after including B dataset
    
    # Now we sample for A and G
    for key in ['A', 'G']:
        df_meta = dfs[key]
        max_images = min(df_meta['Number of Images'].sum(), max_images_per_df)  # Max of 28,000 images or less if fewer available
        max_memory = remaining_memory / len(['A', 'G'])  # Distribute remaining memory dynamically between A and G

        # Use relaxed stratified sampling to maximize images while keeping some stage diversity
        valid_sample, memory_used = relaxed_stratified_sample(df_meta, max_images, max_memory, stage_columns, seed=seed)

        # Store sampled data and adjust remaining memory
        sampled_dfs[key] = valid_sample
        remaining_memory -= memory_used
        
        # After getting the balanced sample, try to add more patients if space remains
        remaining_images = max_images_per_df - sampled_dfs[key]['Number of Images'].sum()
        sampled_dfs[key] = add_more_patients(df_meta, sampled_dfs[key], remaining_images, remaining_memory, seed=seed)

    return sampled_dfs, remaining_memory

'''_______________________________________________________________________________________________________________'''

# Adjust sample sizes if necessary
def adjust_sample_size(sampled_dfs, min_memory_limit_mb, max_memory_limit_mb):
    total_memory_used = sum(df['File Size'].sum() for key, df in sampled_dfs.items())
    
    while total_memory_used < min_memory_limit_mb:
        for key in ['A', 'G']:  # Only adjust for A and G
            if total_memory_used >= max_memory_limit_mb:
                break
            df_meta = dfs[key]
            additional_sample = df_meta[~df_meta.index.isin(sampled_dfs[key].index)]
            if additional_sample.empty:
                continue
            additional_sample = additional_sample.sort_values(by='File Size').iloc[:1]
            sampled_dfs[key] = pd.concat([sampled_dfs[key], additional_sample])
            total_memory_used += additional_sample['File Size'].sum()
        
    while total_memory_used > max_memory_limit_mb:
        for key in ['A', 'G']:  # Only adjust for A and G
            if total_memory_used <= min_memory_limit_mb:
                break
            df_meta = sampled_dfs[key]
            if len(df_meta) > 1:
                # Use iloc instead of index to drop the largest file size row
                largest_file_index = df_meta['File Size'].idxmax()
                largest_file_pos = df_meta['File Size'].sort_values(ascending=False).index[0]
                total_memory_used -= df_meta.loc[largest_file_pos, 'File Size']
                df_meta = df_meta.drop(largest_file_pos)
                sampled_dfs[key] = df_meta
    
    return sampled_dfs, total_memory_used

'''_______________________________________________________________________________________________________________'''

# Function to calculate the distribution of stage columns
def calculate_stage_distribution(df, stage_columns):
    # Group by the stage columns and count the occurrences of each combination
    distribution = df.groupby(stage_columns).size().reset_index(name='count')
    
    # Calculate the proportions (percentage) for each combination
    distribution['proportion'] = distribution['count'] / distribution['count'].sum()
    
    return distribution

'''_______________________________________________________________________________________________________________'''

# Function to compare distributions between the original and sampled datasets
def compare_distributions(original_df, sampled_df, stage_columns):
    # Calculate the distribution for the original and sampled data
    original_distribution = calculate_stage_distribution(original_df, stage_columns)
    sampled_distribution = calculate_stage_distribution(sampled_df, stage_columns)

    # Merge the two distributions on the stage columns
    comparison = pd.merge(original_distribution, sampled_distribution, on=stage_columns, suffixes=('_original', '_sampled'), how='outer').fillna(0)

    # Compute the absolute difference in proportions
    comparison['proportion_diff'] = abs(comparison['proportion_original'] - comparison['proportion_sampled'])
    
    # Compute the relative difference in proportions (percentage difference)
    comparison['relative_diff_percentage'] = (comparison['proportion_diff'] / comparison['proportion_original']) * 100

    return comparison

'''_______________________________________________________________________________________________________________'''

# Function to compare the balance of all datasets against the original
def check_balance(dfs, final_samples, stage_columns):
    for key in dfs.keys():
        print(f"\n--- Balance Comparison for Dataset {key} ---")
        
        # Compare the original dataset with the sampled one
        comparison = compare_distributions(dfs[key], final_samples[key], stage_columns)
        
        # Display the comparison
        display(comparison[['N-Stage', 'Ｍ-Stage', 'T-Stage', 'Histopathological grading', 'proportion_original', 'proportion_sampled', 'relative_diff_percentage']])


'''_______________________________________________________________________________________________________________'''

def read_dicom_image(image_path):
    dicom = pydicom.dcmread(image_path)
    img = dicom.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    return img

def getUID_path(path):
    uid_dict = {}
    for patient_folder in sorted(os.listdir(path)):
        patient_path = os.path.join(path, patient_folder)
        if os.path.isdir(patient_path):
            for root, dirs, files in sorted(os.walk(patient_path)):
                for file in sorted(files):
                    if file.endswith('.dcm'):
                        dicom_path = os.path.join(root, file)
                        try:
                            info = loadFileInformation(dicom_path)
                            uid = info['dicom_num']
                            patient_id = patient_folder.replace('Lung_Dx-', '')
                            description = os.path.basename(os.path.dirname(os.path.dirname(dicom_path)))
                            subdescription = os.path.basename(os.path.dirname(dicom_path))
                            uid_dict[uid] = (dicom_path, patient_id, description, subdescription, file)
                        except Exception as e:
                            print(f"Error processing file {dicom_path}: {e}")
    return dict(sorted(uid_dict.items(), key=lambda item: (item[1][1], item[1][4])))

'''_______________________________________________________________________________________________________________'''


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
'''_______________________________________________________________________________________________________________'''


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

'''_______________________________________________________________________________________________________________'''


def visualize_image_with_bboxes(image, bboxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
    
'''_______________________________________________________________________________________________________________'''

def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

'''_______________________________________________________________________________________________________________'''

def patient_count_by_group(dataset):
    group_image_counts = defaultdict(int)
    group_patient_counts = defaultdict(set)  # Using set to keep unique patient IDs
    
    for image, bboxes, patient_id, description, subdescription, filename in dataset:
        group = patient_id[0]  # First letter from the patient's ID
        group_image_counts[group] += 1
        group_patient_counts[group].add(patient_id)
    
    # Calculate the number of unique patients per group
    unique_patient_counts = {group: len(patients) for group, patients in group_patient_counts.items()}
    
    return group_image_counts, unique_patient_counts

'''_______________________________________________________________________________________________________________'''

def extract_bounding_boxes(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes

'''_______________________________________________________________________________________________________________'''

def read_dicom_image(image_path):
    dicom = pydicom.dcmread(image_path)
    img = dicom.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    return img

'''_______________________________________________________________________________________________________________'''

def visualize_image_by_uid(dataset, target_uid):
    """
    Visualizes an image and its bounding boxes from the dataset based on a specific UID.

    Args:
        dataset (list): The dataset containing image data and metadata.
        target_uid (str): The UID of the image to visualize.
    """
    # Search for the entry with the specified UID
    entry = next((item for item in dataset if item["Uid"] == target_uid), None)
    
    if entry is None:
        print(f"No entry found for UID: {target_uid}")
        return

    # Extract image and bounding boxes
    image = entry["Image"]
    bboxes = entry["BoundingBoxes"]
    patient_id = entry["Patient ID"]
    description = entry["Description"]
    subdescription = entry["Subdescription"]
    filename = entry["Filename"]

    # Display metadata
    print(f"UID: {target_uid}")
    print(f"Patient ID: {patient_id}")
    print(f"Description: {description}")
    print(f"Subdescription: {subdescription}")
    print(f"Filename: {filename}")
    print(f"Bounding Boxes: {bboxes}")

    # Use the separate visualization function
    visualize_image_with_bboxes(image, bboxes)
    

'''_______________________________________________________________________________________________________________'''

from collections import defaultdict

def count_labels_by_class_and_source(output_dir):
    """
    Counts the number of labels for each class in the labels directory,
    distinguishing between original and augmented labels.

    Args:
        output_dir (str): Output directory containing the labels.

    Returns:
        dict: A nested dictionary with counts for each class, separated into 'original' and 'augmented'.
    """
    label_output_dir = os.path.join(output_dir, "labels")
    class_counts = defaultdict(lambda: {"original": 0, "augmented": 0})

    if not os.path.exists(label_output_dir):
        print(f"Error: Directory {label_output_dir} does not exist.")
        return {}

    # Iterate through label files
    for filename in os.listdir(label_output_dir):
        if filename.endswith(".txt"):  # Check for label files
            label_path = os.path.join(label_output_dir, filename)
            with open(label_path, "r") as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    if parts:  # Ensure the line is not empty
                        class_id = int(parts[0])  # Class ID is the first number in YOLO format
                        
                        # Determine if the file is original or augmented based on its name
                        if "_aug_" in filename:
                            class_counts[class_id]["augmented"] += 1
                        else:
                            class_counts[class_id]["original"] += 1

    return dict(class_counts)


'''_______________________________________________________________________________________________________________'''

def count_labels_by_class(output_dir):
    """
    Counts the number of labels for each class in the labels directory.

    Args:
        output_dir (str): Output directory containing the images and labels.

    Returns:
        dict: A dictionary with class IDs as keys and counts as values.
    """
    label_output_dir = os.path.join(output_dir, "labels")
    class_counts = defaultdict(int)

    if not os.path.exists(label_output_dir):
        print(f"Error: Directory {label_output_dir} does not exist.")
        return {}

    # Iterate through label files
    for filename in os.listdir(label_output_dir):
        if filename.endswith(".txt"):  # Check for label files
            label_path = os.path.join(label_output_dir, filename)
            with open(label_path, "r") as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    if parts:  # Ensure the line is not empty
                        class_id = int(parts[0])  # Class ID is the first number in YOLO format
                        class_counts[class_id] += 1

    return dict(class_counts)
'''_______________________________________________________________________________________________________________'''

def augment_yolo_images_train(input_dir, output_dir, target_samples, target_size=(512, 512)):
    """
    Augments images and YOLO format annotations to achieve the target number of samples,
    while keeping the originals in the output directory and maintaining a structured naming convention.

    Args:
        input_dir (str): Directory containing images and labels subdirectories.
        output_dir (str): Directory to save original and augmented images and annotations.
        target_samples (int): Target number of samples per class.
        target_size (tuple): Target image size (width, height).

    Returns:
        None
    """
    # Directories
    image_dir = os.path.join(input_dir, "images")
    label_dir = os.path.join(input_dir, "labels")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise ValueError("Input directory must contain 'images' and 'labels' subdirectories.")

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Augmentation pipelines
    augmentation_pipeline = Compose([
        HorizontalFlip(p=0.05),
        RandomBrightnessContrast(brightness_limit=(0.02, 0.05), contrast_limit=(0.02, 0.05), p=0.3),
        Affine(translate_percent=(0.01, 0.02), scale=(0.95, 1.05), rotate=(-2, 2), p=0.6),
        GaussianBlur(blur_limit=(1, 3), p=0.3),
        CLAHE(clip_limit=(1.5, 2.0), p=0.2),
        RandomGamma(gamma_limit=(90, 110), p=0.2),
    ], bbox_params={'format': 'yolo', 'label_fields': ['category_ids']})

    intense_augmentation_pipeline = Compose([
        HorizontalFlip(p=0.3),
        RandomBrightnessContrast(brightness_limit=(0.1, 0.2), contrast_limit=(0.1, 0.2), p=0.5),
        Affine(translate_percent=(0.05, 0.1), scale=(0.8, 1.2), rotate=(-10, 10), shear=(-5, 5), p=0.8),
        GaussianBlur(blur_limit=(1, 7), p=0.6),
        CLAHE(clip_limit=(1.0, 5.0), p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.4),
    ], bbox_params={'format': 'yolo', 'label_fields': ['category_ids']})

    # Collect image and label paths
    image_paths = glob(os.path.join(image_dir, '*.jpg'))
    label_paths = [
        os.path.join(label_dir, os.path.basename(img).replace('.jpg', '.txt'))
        for img in image_paths
    ]

    # Group images by class based on label files
    class_images = {}
    for img_path, lbl_path in zip(image_paths, label_paths):
        with open(lbl_path, "r") as file:
            lines = file.readlines()
        if not lines:
            continue
        class_id = int(lines[0].split()[0])
        if class_id not in class_images:
            class_images[class_id] = []
        class_images[class_id].append((img_path, lbl_path))

    # Count initial samples per class
    class_counts = {class_id: len(images) for class_id, images in class_images.items()}

    # Copy original images and labels to output directory
    for class_id, images in class_images.items():
        for img_path, lbl_path in images:
            original_image_out = os.path.join(output_dir, "images", os.path.basename(img_path))
            original_label_out = os.path.join(output_dir, "labels", os.path.basename(lbl_path))
            if not os.path.exists(original_image_out):
                shutil.copy(img_path, original_image_out)
            if not os.path.exists(original_label_out):
                shutil.copy(lbl_path, original_label_out)

    # Augment images to meet target samples
    for class_id, images in class_images.items():
        if not images:
            print(f"No valid images found for Class {class_id} to augment.")
            continue

        images = images[:]
        random.shuffle(images)
        augment_index = 1

        while class_counts[class_id] < int(target_samples):
            for img_path, lbl_path in images:
                # Read the image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                # Read bounding boxes
                with open(lbl_path, "r") as file:
                    yolo_bboxes = [list(map(float, line.split()[1:])) for line in file.readlines()]

                # Select appropriate pipeline
                pipeline = intense_augmentation_pipeline if os.path.basename(img_path).startswith(("A","E", "B", "G")) else augmentation_pipeline

                # Apply augmentations
                augmented = pipeline(
                    image=image,
                    bboxes=yolo_bboxes,
                    category_ids=[class_id] * len(yolo_bboxes)
                )
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']

                # Normalize the augmented image to [0, 1]
                augmented_image = augmented_image / 255.0

                # Save the normalized augmented image
                base_filename = os.path.basename(img_path).replace('.jpg', '')
                aug_filename = f"{base_filename}_aug_{augment_index}.jpg"
                aug_image_path = os.path.join(output_dir, "images", aug_filename)
                cv2.imwrite(aug_image_path, cv2.cvtColor((augmented_image * 255).astype('uint8'), cv2.COLOR_RGB2BGR))

                # Save YOLO annotations
                aug_label_path = os.path.join(output_dir, "labels", aug_filename.replace('.jpg', '.txt'))
                with open(aug_label_path, "w") as file:
                    for bbox in augmented_bboxes:
                        bbox_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in bbox)
                        file.write(bbox_line + "\n")

                class_counts[class_id] += 1
                augment_index += 1

                if class_counts[class_id] >= int(target_samples):
                    break

            random.shuffle(images)

    print("\nAugmentation completed.")
    for class_id, count in class_counts.items():
        print(f"  Total images for Class {class_id} after augmentation: {count}")
    
    
'''_______________________________________________________________________________________________________________'''

def augment_yolo_images_val(input_dir, output_dir, target_samples, target_size=(512, 512)):
    """
    Augments images and YOLO format annotations to achieve the target number of samples,
    while keeping the originals in the output directory and maintaining a structured naming convention.

    Args:
        input_dir (str): Directory containing `images` and `labels` subdirectories.
        output_dir (str): Directory to save original and augmented images and annotations.
        target_samples (int): Target number of samples per class.
        target_size (tuple): Target image size (width, height).

    Returns:
        None
    """
    # Directories
    image_dir = os.path.join(input_dir, "images")
    label_dir = os.path.join(input_dir, "labels")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise ValueError("Input directory must contain 'images' and 'labels' subdirectories.")

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Augmentation pipeline
    augmentation_pipeline = Compose([
        HorizontalFlip(p=0.05),
        RandomBrightnessContrast(brightness_limit=(0.03, 0.05), contrast_limit=(0.03, 0.05), p=0.3),
        Affine(translate_percent=(0.01, 0.02), scale=(0.95, 1.05), rotate=(-2, 2), p=0.6),
        GaussianBlur(blur_limit=(1, 3), p=0.3),
        CLAHE(clip_limit=(1.5, 2.0), p=0.2),
        RandomGamma(gamma_limit=(90, 110), p=0.1),
    ], bbox_params={'format': 'yolo', 'label_fields': ['category_ids']})

    # Collect image and label paths
    image_paths = glob(os.path.join(image_dir, '*.jpg'))
    label_paths = [
        os.path.join(label_dir, os.path.basename(img).replace('.jpg', '.txt'))
        for img in image_paths
    ]

    # Group images by class based on label files
    class_images = {}
    for img_path, lbl_path in zip(image_paths, label_paths):
        with open(lbl_path, "r") as file:
            lines = file.readlines()
        if not lines:
            continue
        class_id = int(lines[0].split()[0])
        if class_id not in class_images:
            class_images[class_id] = []
        class_images[class_id].append((img_path, lbl_path))

    class_counts = {class_id: len(images) for class_id, images in class_images.items()}

    # Copy original images and labels to output directory
    for class_id, images in class_images.items():
        for img_path, lbl_path in images:
            original_image_out = os.path.join(output_dir, "images", os.path.basename(img_path))
            original_label_out = os.path.join(output_dir, "labels", os.path.basename(lbl_path))
            if not os.path.exists(original_image_out):
                shutil.copy(img_path, original_image_out)
            if not os.path.exists(original_label_out):
                shutil.copy(lbl_path, original_label_out)

    # Augment images to meet the target number of samples
    for class_id, images in class_images.items():
        if not images:
            print(f"No valid images found for Class {class_id} to augment.")
            continue

        images = images[:]
        random.shuffle(images)
        augment_index = 1

        while class_counts[class_id] < int(target_samples):
            for img_path, lbl_path in images:
                # Read the image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                # Read bounding boxes
                with open(lbl_path, "r") as file:
                    yolo_bboxes = [list(map(float, line.split()[1:])) for line in file.readlines()]

                # Apply augmentations
                augmented = augmentation_pipeline(
                    image=image,
                    bboxes=yolo_bboxes,
                    category_ids=[class_id] * len(yolo_bboxes)
                )
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']

                # Normalize the augmented image to [0, 1]
                augmented_image = augmented_image / 255.0

                # Save the normalized augmented image
                base_filename = os.path.basename(img_path).replace('.jpg', '')
                aug_filename = f"{base_filename}_aug_{augment_index}.jpg"
                aug_image_path = os.path.join(output_dir, "images", aug_filename)
                cv2.imwrite(aug_image_path, cv2.cvtColor((augmented_image * 255).astype('uint8'), cv2.COLOR_RGB2BGR))

                # Save YOLO annotations
                aug_label_path = os.path.join(output_dir, "labels", aug_filename.replace('.jpg', '.txt'))
                with open(aug_label_path, "w") as file:
                    for bbox in augmented_bboxes:
                        bbox_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in bbox)
                        file.write(bbox_line + "\n")

                class_counts[class_id] += 1
                augment_index += 1

                if class_counts[class_id] >= int(target_samples):
                    break

            random.shuffle(images)

    print("\nAugmentation completed.")
    for class_id, count in class_counts.items():
        print(f"  Total images for Class {class_id} after augmentation: {count}")


'''_______________________________________________________________________________________________________________'''
    
def visualize_yolo_images(output_dir, class_mapping, num_images=5, target_size=(512, 512)):
    """
    Visualizes random images with bounding boxes and prints the UID and filename in the console.

    Args:
        output_dir (str): The directory containing the processed images and labels.
        class_mapping (dict): The mapping of class names to IDs.
        num_images (int): Number of random images to visualize.
        target_size (tuple): Target size of images (width, height).
    """
    image_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print("Error: Image or label directories do not exist.")
        return

    reverse_mapping = {v: k for k, v in class_mapping.items()}  # Reverse mapping for class name lookup

    # Collect all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    if len(image_files) < num_images:
        print(f"Not enough images to display. Found {len(image_files)}.")
        num_images = len(image_files)

    # Select random images
    random_files = random.sample(image_files, num_images)

    # Display the selected images
    for image_file in random_files:
        uid = image_file.split("_")[0]
        label_file = os.path.splitext(image_file)[0] + ".txt"

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and overlay bounding boxes
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    class_id, x_center, y_center, width, height = map(float, parts)
                    class_id = int(class_id)

                    x_center *= target_size[0]
                    y_center *= target_size[1]
                    width *= target_size[0]
                    height *= target_size[1]

                    xmin = int(x_center - width / 2)
                    ymin = int(y_center - height / 2)
                    xmax = int(x_center + width / 2)
                    ymax = int(y_center + height / 2)

                    # Draw bounding box (red)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Red in BGR
                    cv2.putText(image, f"Class {class_id} ({reverse_mapping[class_id]})", 
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Smaller label

        # Print UID and image filename to console
        print(f"UID: {uid}, Image: {image_file}")

        # Display the image with smaller size
        plt.figure(figsize=(5, 5))  # Smaller display size
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Class: {reverse_mapping[class_id]} (ID: {class_id})")
        plt.show()

        
'''_______________________________________________________________________________________________________________'''
        

def split_data(input_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Splits preprocessed data into train, validation, and test sets based on the total number of images,
    while ensuring all images from a single patient are in the same set. Prioritizes training set diversity.

    Args:
        input_dir (str): Directory containing preprocessed images and labels.
        output_dir (str): Output directory for train, validation, and test splits.
        train_ratio (float): Proportion of data to be used for training.
        val_ratio (float): Proportion of data to be used for validation.
        test_ratio (float): Proportion of data to be used for testing.

    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum up to 1."

    image_dir = os.path.join(input_dir, "images")
    label_dir = os.path.join(input_dir, "labels")

    # Group images by patient ID and class (first digit of the patient ID)
    patient_to_files = defaultdict(list)
    patient_to_class = {}

    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg"):
            patient_id = image_file[:5]  # Extract the first 5 digits as patient ID
            patient_class = image_file[0]  # First digit represents the class
            label_file = image_file.replace(".jpg", ".txt")
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)

            if os.path.exists(label_path):
                patient_to_files[patient_id].append((image_path, label_path))
                patient_to_class[patient_id] = patient_class

    # Group patients by class
    class_to_patients = defaultdict(list)
    for patient_id, patient_class in patient_to_class.items():
        class_to_patients[patient_class].append(patient_id)

    # Shuffle patients within each class for randomness
    for patient_list in class_to_patients.values():
        shuffle(patient_list)

    # Split patients into train, validation, and test sets while ensuring specific handling for class E
    train_patients, val_patients, test_patients = set(), set(), set()

    # Explicitly allocate patients from class E
    if 'E' in class_to_patients:
        e_patients = class_to_patients['E']
        train_patients.update(['E0001', 'E0002', 'E0004'])  # Explicitly assign patients to train
        val_patients.add('E0003')  # Assign E0003 to validation
        test_patients.add('E0005')  # Assign E0005 to test
        del class_to_patients['E']  # Remove class E from further processing

    for patient_class, patients in class_to_patients.items():
        # General splitting for other classes
        num_patients = len(patients)
        train_end = int(num_patients * train_ratio)
        val_end = train_end + int(num_patients * val_ratio)

        train_patients.update(patients[:train_end])
        val_patients.update(patients[train_end:val_end])
        test_patients.update(patients[val_end:])

    # Ensure no overlap between sets
    assert train_patients.isdisjoint(val_patients), "Train and validation sets overlap."
    assert train_patients.isdisjoint(test_patients), "Train and test sets overlap."
    assert val_patients.isdisjoint(test_patients), "Validation and test sets overlap."

    # Adjust validation and test sets to ensure balanced image ratios per class
    for class_label, patients in class_to_patients.items():
        val_images = sum(len(patient_to_files[p]) for p in val_patients if p in patients)
        test_images = sum(len(patient_to_files[p]) for p in test_patients if p in patients)

        total_images = sum(len(patient_to_files[p]) for p in patients)
        target_val_images = int(total_images * val_ratio)
        target_test_images = int(total_images * test_ratio)

        while val_images < target_val_images:
            candidate = next((p for p in train_patients if p in patients), None)
            if candidate:
                train_patients.remove(candidate)
                val_patients.add(candidate)
                val_images += len(patient_to_files[candidate])

        while test_images < target_test_images:
            candidate = next((p for p in train_patients if p in patients), None)
            if candidate:
                train_patients.remove(candidate)
                test_patients.add(candidate)
                test_images += len(patient_to_files[candidate])

        # Adjust if validation set exceeds its target
        while val_images > target_val_images:
            candidate = next((p for p in val_patients if p in patients), None)
            if candidate:
                val_patients.remove(candidate)
                train_patients.add(candidate)
                val_images -= len(patient_to_files[candidate])

        # Adjust if test set exceeds its target
        while test_images > target_test_images:
            candidate = next((p for p in test_patients if p in patients), None)
            if candidate:
                test_patients.remove(candidate)
                train_patients.add(candidate)
                test_images -= len(patient_to_files[candidate])

    # Collect files for each set
    train_files = [file for patient in train_patients for file in patient_to_files[patient]]
    val_files = [file for patient in val_patients for file in patient_to_files[patient]]
    test_files = [file for patient in test_patients for file in patient_to_files[patient]]

    # Helper function to copy files
    def copy_files(file_list, subset_name):
        subset_image_dir = os.path.join(output_dir, subset_name, "images")
        subset_label_dir = os.path.join(output_dir, subset_name, "labels")
        os.makedirs(subset_image_dir, exist_ok=True)
        os.makedirs(subset_label_dir, exist_ok=True)

        for image_path, label_path in file_list:
            shutil.copy(image_path, subset_image_dir)
            shutil.copy(label_path, subset_label_dir)

    # Copy files to their respective directories
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print("Data split completed:")
    print(f"  Train set: {len(train_files)} samples ({len(train_patients)} patients)")
    print(f"  Validation set: {len(val_files)} samples ({len(val_patients)} patients)")
    print(f"  Test set: {len(test_files)} samples ({len(test_patients)} patients)")
    
'''_______________________________________________________________________________________________________________'''


def count_images_labels_patients_by_class(output_directory):
    def analyze_folder(folder):
        counts_by_letter = defaultdict(int)
        unique_patient_ids = set()

        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)):
                # Count files starting with specific letters
                first_letter = file[0].upper()
                if first_letter in ['A', 'B', 'E', 'G']:
                    counts_by_letter[first_letter] += 1

                # Extract and store unique patient IDs
                patient_id = file[:5]  # Assuming the first five characters are the patient ID
                unique_patient_ids.add(patient_id)

        return counts_by_letter, unique_patient_ids

    # Define paths for images and labels folders
    images_folder = os.path.join(output_directory, "images")
    labels_folder = os.path.join(output_directory, "labels")

    # Check if the folders exist
    if not os.path.isdir(images_folder):
        raise FileNotFoundError(f"Images folder not found in {output_directory}")
    if not os.path.isdir(labels_folder):
        raise FileNotFoundError(f"Labels folder not found in {output_directory}")

    # Analyze images folder
    images_counts, images_patient_ids = analyze_folder(images_folder)

    # Analyze labels folder
    labels_counts, labels_patient_ids = analyze_folder(labels_folder)

    # Combine results
    class_counts = {
        letter: {
            "images": images_counts.get(letter, 0),
            "labels": labels_counts.get(letter, 0),
            "patients": len(set(filter(lambda x: x[0] == letter, images_patient_ids.union(labels_patient_ids))))
        } for letter in set(images_counts.keys()).union(labels_counts.keys())
    }

    patient_ids = {
        letter: sorted(set(filter(lambda x: x[0] == letter, images_patient_ids.union(labels_patient_ids))))
        for letter in set(images_counts.keys()).union(labels_counts.keys())
    }

    return class_counts, patient_ids

'''_______________________________________________________________________________________________________________'''


def preprocess_images(image_dir, annotation_dir, output_dir, target_size=(512, 512)):
    """
    Preprocesses all images and saves them with annotations in YOLO format, only for images with corresponding bounding boxes.

    Args:
        image_dir (str): Directory with DICOM images organized by patient folders.
        annotation_dir (str): Directory with XML annotations.
        output_dir (str): Output directory for YOLO dataset.
        target_size (tuple): Target image size (width, height).

    Returns:
        dict: Dictionary of processed images organized by class.
    """
    class_mapping = {
        "A": 0,  # Adenocarcinoma
        "B": 1,  # Small Cell Carcinoma
        "E": 2,  # Large Cell Carcinoma
        "G": 3   # Squamous Cell Carcinoma
    }

    # Prepare output directories
    image_output_dir = os.path.join(output_dir, "images")
    label_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    class_images = defaultdict(list)
    uid_dict = getUID_path(image_dir)

    for uid, (image_path, patient_id, description, subdescription, filename) in uid_dict.items():
        for root, dirs, files in sorted(os.walk(annotation_dir)):
            for file in sorted(files):
                if file == f'{uid}.xml':
                    annotation_path = os.path.join(root, file)

                    # Extract cancer type from patient folder name
                    cancer_type = str(patient_id[0])
                    if cancer_type not in class_mapping:
                        continue

                    class_id = class_mapping[cancer_type]

                    # Read and preprocess image
                    try:
                        image = read_dicom_image(image_path)
                    except Exception:
                        continue

                    # Extract bounding boxes
                    bboxes = extract_bounding_boxes(annotation_path)
                    if not bboxes:
                        # Skip images without bounding boxes
                        continue

                    original_height, original_width = image.shape[:2]
                    resized_image = cv2.resize(image, target_size)
                    rescaled_bboxes = [
                        [
                            int(bbox[0] * target_size[0] / original_width),
                            int(bbox[1] * target_size[1] / original_height),
                            int(bbox[2] * target_size[0] / original_width),
                            int(bbox[3] * target_size[1] / original_height)
                        ]
                        for bbox in bboxes
                    ]

                    # Save preprocessed image and labels
                    image_filename = f"{patient_id}_{uid}.jpg"
                    image_path_out = os.path.join(image_output_dir, image_filename)
                    cv2.imwrite(image_path_out, resized_image)

                    yolo_bboxes = []
                    for bbox in rescaled_bboxes:
                        xmin, ymin, xmax, ymax = bbox
                        x_center = ((xmin + xmax) / 2) / target_size[0]
                        y_center = ((ymin + ymax) / 2) / target_size[1]
                        width = (xmax - xmin) / target_size[0]
                        height = (ymax - ymin) / target_size[1]
                        yolo_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                    label_filename = f"{patient_id}_{uid}.txt"
                    label_path_out = os.path.join(label_output_dir, label_filename)
                    with open(label_path_out, "w") as label_file:
                        label_file.write("\n".join(yolo_bboxes))

                    class_images[class_id].append((resized_image, rescaled_bboxes, class_id, uid, patient_id))
                    break

    return class_images


'''_______________________________________________________________________________________________________________'''


def visualize_image_with_bboxes_legend(image, bboxes, patient_id):
    """
    Visualizes an image with bounding boxes and displays a legend on top of each box.

    Args:
        image (ndarray): The image to visualize.
        bboxes (list): A list of bounding boxes, where each box is [xmin, ymin, xmax, ymax].
        patient_id (str): The patient ID (used to determine the cancer type).
    """
    # Cancer type mapping
    CANCER_TYPE_MAPPING = {
        "A": "Adenocarcinoma",
        "B": "Small Cell Carcinoma",
        "E": "Large Cell Carcinoma",
        "G": "Squamous Cell Carcinoma"
    }
    
    # Get the cancer type based on the first character of the patient ID
    cancer_type_key = patient_id[0]  # First character of patient ID
    cancer_type = CANCER_TYPE_MAPPING.get(cancer_type_key, "Unknown")

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add legend text above the bounding box
        text_x = xmin
        text_y = ymin - 5  # Position text slightly above the bounding box
        ax.text(
            text_x, text_y, f"({cancer_type_key}) {cancer_type}",
            color="white", fontsize=8, bbox=dict(facecolor='red', alpha=0.5)
        )
    
    plt.axis('off')
    plt.show()