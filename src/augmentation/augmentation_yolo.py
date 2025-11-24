# Imports
import os
import shutil
import random
from glob import glob

import cv2
from albumentations import (
    Compose,
    HorizontalFlip,
    RandomBrightnessContrast,
    Affine,
    GaussianBlur,
    CLAHE,
    RandomGamma
)
import numpy as np


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