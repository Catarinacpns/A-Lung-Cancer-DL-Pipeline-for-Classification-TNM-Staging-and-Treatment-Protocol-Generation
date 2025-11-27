import os

#Terminal: python tree.py > repo_structure.txt

# Folders to exclude completely
EXCLUDE_FOLDERS = {
    "raw", "NSCLC_Radiomics", "Annotation", "sampled", "yolo", "trial_results", "TNM", "trial_results_40epochs", "trial_results_SGD_StepLR", "trial_results_SGD_StepLR_50epochs", "trial_results_SGD_StepLR_50epochs_2", "trial_results_SGD","test_YOLO_orig(16batch)", "test_YOLOorig_nano", "test_YOLOorig_nano(16batch)_E_G_2", "test_YOLOorig_nano(16batch)_E_G_last", "test_YOLOorig_nano(32batch)_E_G", "test_YOLOorig_small(16batch)", "test_YOLOorig_small(16batch)_E_G_2", "test_YOLOorig_small(16batch)_E_G_best", "test_YOLOorig_small(32batch)_E_G", "test_lung_cancer_detection_ResNet50", "test_YOLOorig_nano(16batch)_E_G", "__pycache__"
}

# File extensions to exclude
EXCLUDE_EXTENSIONS = {".jpeg", ".dcm", ".nii", ".zip"}

def is_excluded(path):
    # exclude folders
    for folder in EXCLUDE_FOLDERS:
        if f"/{folder}/" in path or path.endswith(folder):
            return True

    # exclude files by extension
    _, ext = os.path.splitext(path)
    if ext.lower() in EXCLUDE_EXTENSIONS:
        return True

    return False


def list_structure(start_path=".", indent=""):
    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        return

    for item in items:
        if item.startswith("."):
            continue

        full_path = os.path.join(start_path, item)

        # Skip excluded files/folders
        if is_excluded(full_path):
            print(indent + "├── " + item + "   [contains files inside]")
            continue

        print(indent + "├── " + item)

        # If directory, recurse
        if os.path.isdir(full_path):
            list_structure(full_path, indent + "│   ")


if __name__ == "__main__":
    print("Repository Structure:\n")
    list_structure(".")
