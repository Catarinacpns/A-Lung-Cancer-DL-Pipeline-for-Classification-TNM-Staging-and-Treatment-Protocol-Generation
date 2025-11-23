import os
import shutil
from collections import defaultdict
from random import shuffle

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
    print(f"  Test set: {len(test_files)} samples ({len(test_patients)} patients)")