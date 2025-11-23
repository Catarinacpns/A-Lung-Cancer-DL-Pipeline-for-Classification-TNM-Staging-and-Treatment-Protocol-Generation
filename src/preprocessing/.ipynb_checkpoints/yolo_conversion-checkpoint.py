import os
import cv2
from collections import defaultdict

# Custom project imports
from src.preprocessing.uid_mapping import getUID_path
from src.preprocessing.dicom_io import read_dicom_image
from src.preprocessing.xml_parsing import extract_bounding_boxes

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