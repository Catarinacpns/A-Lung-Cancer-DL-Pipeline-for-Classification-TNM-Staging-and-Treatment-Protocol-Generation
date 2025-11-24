import os
from collections import defaultdict
from collections import Counter
import re

def count_unique_subject_ids(df, letter):
    filtered = df[df['Subject ID'].str.startswith(letter)]
    unique_count = filtered['Subject ID'].str[1:].nunique()
    return unique_count

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


def get_patient_images(folder_path, prefixes):
    """Maps patients to their images per histology prefix."""
    patient_images = {prefix: defaultdict(list) for prefix in prefixes}

    for filename in os.listdir(folder_path):
        for prefix in prefixes:
            if filename.startswith(prefix):
                patient_id = filename.split("_")[1]  # Extract patient ID
                patient_images[prefix][patient_id].append(filename)

    return patient_images


def get_patient_images_v2(folder_path, prefix):
    """Organiza imagens por paciente para um determinado prefixo."""
    patient_images = defaultdict(list)
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            patient_id = filename.split("_")[1]  # Extrai o ID do paciente
            patient_images[patient_id].append(filename)
    return patient_images


def sample_patients_by_image_count(patient_images, target_images):
    """Selects patients iteratively until the total number of images reaches the target."""
    selected_patients = {prefix: [] for prefix in patient_images.keys()}  # List to store patient IDs
    selected_images = {prefix: [] for prefix in patient_images.keys()}  # List to store images

    for prefix, patients in patient_images.items():
        # Sort patients by number of images (ascending order)
        sorted_patients = sorted(patients.items(), key=lambda x: len(x[1]))

        total_images = 0
        for patient_id, images in sorted_patients:
            if total_images >= target_images:
                break  # Stop when reaching the target
            
            selected_patients[prefix].append((patient_id, len(images)))  # Store patient ID & image count
            selected_images[prefix].extend(images)
            total_images += len(images)

    return selected_patients, selected_images


def sample_patients(patient_images, target_images):
    """Seleciona pacientes até atingir o número desejado de imagens."""
    selected_patients = []
    selected_images = []
    sorted_patients = sorted(patient_images.items(), key=lambda x: len(x[1]))
    total_images = 0
    
    for patient_id, images in sorted_patients:
        if total_images >= target_images:
            break
        selected_patients.append(patient_id)
        selected_images.extend(images)
        total_images += len(images)
    
    return selected_patients, selected_images


def count_files_and_patients(folder_path, prefixes):
    """Counts the number of files and unique patients per histology prefix."""
    file_counts = Counter()
    patient_counts = {prefix: set() for prefix in prefixes}  # Use sets to track unique patients
    
    for filename in os.listdir(folder_path):
        for prefix in prefixes:
            if filename.startswith(prefix):
                file_counts[prefix] += 1
                patient_id = filename.split("_")[1]  # Extract patient ID
                patient_counts[prefix].add(patient_id)

    # Convert patient sets to counts
    patient_counts = {prefix: len(patients) for prefix, patients in patient_counts.items()}
    
    return file_counts, patient_counts

    
    
def extract_patient_id_2datasets(filename):
    """Extracts the patient ID while maintaining the correct format."""
    match = re.match(r"(E_LUNG1-\d+|G_LUNG1-\d+|E\d+|G\d+|A\d+|B\d+)", filename)  
    return match.group() if match else None  # Return matched patient ID
    
def count_images_labels_patients_by_class_total(base_directory):
    def analyze_folder(folder):
        counts_by_class = defaultdict(int)
        unique_patient_ids = defaultdict(set)

        if not os.path.isdir(folder):
            return counts_by_class, unique_patient_ids  # Return empty if folder doesn't exist

        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)):
                patient_id = extract_patient_id_2datasets(file)  # Extract patient ID
                
                if patient_id:
                    first_letter = patient_id[0]  
                    counts_by_class[first_letter] += 1
                    unique_patient_ids[first_letter].add(patient_id)  # Store unique patient IDs

        return counts_by_class, unique_patient_ids

    # Define dataset splits
    splits = ["train", "val", "test"]
    total_counts = defaultdict(lambda: {"images": 0, "labels": 0, "patients": set()})

    for split in splits:
        images_folder = os.path.join(base_directory, split, "images")
        labels_folder = os.path.join(base_directory, split, "labels")

        # Analyze images and labels per split
        images_counts, images_patient_ids = analyze_folder(images_folder)
        labels_counts, labels_patient_ids = analyze_folder(labels_folder)

        # Aggregate counts across all splits
        for letter in ['A', 'B', 'E', 'G']:
            total_counts[letter]["images"] += images_counts.get(letter, 0)
            total_counts[letter]["labels"] += labels_counts.get(letter, 0)
            total_counts[letter]["patients"].update(images_patient_ids[letter].union(labels_patient_ids[letter]))

    # Convert patient sets to counts
    for letter in ['A', 'B', 'E', 'G']:
        total_counts[letter]["patients"] = len(total_counts[letter]["patients"])

    return total_counts


def count_images_labels_patients_by_class_2datasets(output_directory):
    def analyze_folder(folder):
        counts_by_class = defaultdict(int)
        unique_patient_ids = defaultdict(set)
        dataset_patient_counts = defaultdict(lambda: defaultdict(set))  # {class: {dataset_type: {patients}}}

        if not os.path.isdir(folder):
            return counts_by_class, unique_patient_ids, dataset_patient_counts  # Return empty if folder doesn't exist

        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)):
                patient_id = extract_patient_id_2datasets(file)  # Extract patient ID
                
                if patient_id:
                    class_label = patient_id[0]  # First letter (A, B, E, G)
                    dataset_label = "LUNG1" if "LUNG1" in patient_id else "Standard"  # Separate E_LUNG1 and G_LUNG1
                    
                    counts_by_class[class_label] += 1
                    unique_patient_ids[class_label].add(patient_id)  # Store unique patient IDs
                    dataset_patient_counts[class_label][dataset_label].add(patient_id)  # Store by dataset type

        return counts_by_class, unique_patient_ids, dataset_patient_counts

    # Define paths for images and labels folders
    images_folder = os.path.join(output_directory, "images")
    labels_folder = os.path.join(output_directory, "labels")

    if not os.path.isdir(images_folder):
        raise FileNotFoundError(f"Images folder not found in {output_directory}")
    if not os.path.isdir(labels_folder):
        raise FileNotFoundError(f"Labels folder not found in {output_directory}")

    # Analyze images and labels
    images_counts, images_patient_ids, images_dataset_counts = analyze_folder(images_folder)
    labels_counts, labels_patient_ids, labels_dataset_counts = analyze_folder(labels_folder)

    # Combine results
    class_counts = {}
    patient_ids = {}

    for letter in set(images_counts.keys()).union(labels_counts.keys()):
        class_counts[letter] = {
            "images": images_counts.get(letter, 0),
            "labels": labels_counts.get(letter, 0),
            "total_patients": len(images_patient_ids[letter].union(labels_patient_ids[letter])),
            "patients_standard": len(images_dataset_counts[letter]["Standard"].union(labels_dataset_counts[letter]["Standard"])),
            "patients_lung1": len(images_dataset_counts[letter]["LUNG1"].union(labels_dataset_counts[letter]["LUNG1"]))
        }

        patient_ids[letter] = {
            "all": sorted(images_patient_ids[letter].union(labels_patient_ids[letter])),
            "standard": sorted(images_dataset_counts[letter]["Standard"].union(labels_dataset_counts[letter]["Standard"])),
            "lung1": sorted(images_dataset_counts[letter]["LUNG1"].union(labels_dataset_counts[letter]["LUNG1"]))
        }

    return class_counts, patient_ids

