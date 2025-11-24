import os
import cv2
import matplotlib.pyplot as plt

from .rtstruct_to_yolo import draw_bboxes  
from .rtstruct_io import read_dicom_image  


def visualize_bboxes(ct_slices, bboxes):
    """Visualizes bounding boxes on CT slices while maintaining original grayscale contrast."""
    for z_idx, x_min, y_min, x_max, y_max in bboxes:
        image = read_dicom_image(ct_slices[z_idx])  # Normalize without altering contrast
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.title(f"Slice {z_idx} - Tumor Bounding Box")
        plt.show()
        

def display_all_images_for_patient(image_dir, label_dir, patient_id):
    """Displays all annotated images for a selected patient."""
    patient_images = sorted([f for f in os.listdir(image_dir) if f.startswith(patient_id) and f.endswith(".jpg")])
    
    if not patient_images:
        print(f"No images found for patient {patient_id}")
        return
    
    for image_filename in patient_images:
        image_path = os.path.join(image_dir, image_filename)
        label_filename = image_filename.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_filename)
        
        annotated_image = draw_bboxes(image_path, label_path)
        if annotated_image is not None:
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Patient: {patient_id} - {image_filename}")
            plt.axis("off")
            plt.show()
            
