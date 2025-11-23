import os
import random

import cv2
import matplotlib.pyplot as plt


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