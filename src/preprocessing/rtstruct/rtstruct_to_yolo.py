import os
import cv2


def save_yolo_format(patient_id, study_folder_name, bboxes, output_label_dir):
    """Saves bounding boxes in YOLO format directly in the label folder"""
    os.makedirs(output_label_dir, exist_ok=True)
    for z_idx, x_center, y_center, width, height in bboxes:
        label_filename = os.path.join(output_label_dir, f"{patient_id}_{study_folder_name}_{z_idx}.txt")
        with open(label_filename, "w") as f:
            f.write(f"0 {x_center} {y_center} {width} {height}\n")
            
def save_images_as_jpeg(patient_id, study_folder_name, ct_slices, bboxes, output_image_dir):
    """Saves corresponding CT slices as JPEG images directly in the image folder"""
    os.makedirs(output_image_dir, exist_ok=True)
    for z_idx, _, _, _, _ in bboxes:
        image = read_dicom_image(ct_slices[z_idx])  # Maintain original contrast
        image_filename = os.path.join(output_image_dir, f"{patient_id}_{study_folder_name}_{z_idx}.jpg")
        cv2.imwrite(image_filename, image)

        
def load_yolo_labels(label_path):
    """Loads YOLO format bounding boxes from a text file."""
    bboxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_center, y_center, width, height = map(float, parts)
                bboxes.append((x_center, y_center, width, height))
    return bboxes


def draw_bboxes(image_path, label_path):
    """Loads image and labels, then draws bounding boxes."""
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    if not os.path.exists(label_path):
        return None  # Skip if label file does not exist
    
    bboxes = load_yolo_labels(label_path)

    # Draw bounding boxes
    for x_center, y_center, width, height in bboxes:
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return image