import os
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