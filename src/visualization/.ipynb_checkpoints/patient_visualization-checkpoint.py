from src.visualization.plot_bboxes import visualize_image_with_bboxes

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