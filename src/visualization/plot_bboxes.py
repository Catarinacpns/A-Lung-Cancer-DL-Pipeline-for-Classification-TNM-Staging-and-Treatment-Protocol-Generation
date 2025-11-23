import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image_with_bboxes(image, bboxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
    

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